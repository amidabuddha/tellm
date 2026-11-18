//! Per-reply cost: turns the token usage a provider reports into the cost line
//! shown under each reply, and keeps the running total of a room's current
//! conversation for `/cost`.
//!
//! A cost the provider reports (xAI, OpenRouter) is what was billed and wins.
//! Otherwise the model's `price` table from config.toml gives an estimate,
//! marked with "≈". Without either, the line shows token counts.

use tellm_config::ModelPrice;
use tellm_core::Usage;

/// Where a reply's cost came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CostSource {
    /// The provider reported what it charged.
    Reported,
    /// tellm applied the model's configured prices to the reported usage.
    Estimated,
}

/// One reply's usage and, when it can be known, its cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReplyCost {
    pub(crate) usage: Option<Usage>,
    /// US dollars and where the figure came from; `None` when the provider
    /// reported no cost and the model has no price.
    pub(crate) cost: Option<(f64, CostSource)>,
}

impl ReplyCost {
    pub(crate) fn new(usage: Option<Usage>, price: Option<&ModelPrice>) -> Self {
        let reported = usage
            .and_then(|usage| usage.reported_cost_usd)
            .map(|usd| (usd, CostSource::Reported));
        let cost = reported.or_else(|| {
            usage
                .zip(price)
                .map(|(usage, price)| (price.cost_usd(&usage), CostSource::Estimated))
        });
        Self { usage, cost }
    }
}

/// Running totals for a room's current conversation. They start over
/// whenever the conversation does: `/new`, `/model`, `/role`, `/mode message`,
/// and the terminal `reset`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ChatCost {
    pub(crate) replies: u64,
    /// Replies whose cost is known, reported or estimated.
    pub(crate) costed_replies: u64,
    /// Whether any summed cost is tellm's estimate rather than a reported one.
    pub(crate) estimated: bool,
    /// Sum of the known reply costs, in billionths of a US dollar, so the
    /// totals stay exact integers.
    nano_usd: u64,
    pub(crate) input_tokens: u64,
    pub(crate) cached_input_tokens: u64,
    pub(crate) cache_write_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) reasoning_tokens: u64,
    pub(crate) web_searches: u64,
}

const NANO_USD_PER_DOLLAR: f64 = 1_000_000_000.0;

impl ChatCost {
    pub(crate) fn record(&mut self, reply: &ReplyCost) {
        self.replies = self.replies.saturating_add(1);
        if let Some(usage) = &reply.usage {
            self.input_tokens = self.input_tokens.saturating_add(usage.input_tokens);
            self.cached_input_tokens = self
                .cached_input_tokens
                .saturating_add(usage.cached_input_tokens);
            self.cache_write_tokens = self
                .cache_write_tokens
                .saturating_add(usage.cache_write_tokens);
            self.output_tokens = self.output_tokens.saturating_add(usage.output_tokens);
            self.reasoning_tokens = self.reasoning_tokens.saturating_add(usage.reasoning_tokens);
            self.web_searches = self.web_searches.saturating_add(usage.web_searches);
        }
        if let Some((usd, source)) = reply.cost {
            self.costed_replies = self.costed_replies.saturating_add(1);
            self.estimated |= source == CostSource::Estimated;
            // Float-to-int casts saturate, and NaN becomes 0.
            let nano = (usd.max(0.0) * NANO_USD_PER_DOLLAR).round() as u64;
            self.nano_usd = self.nano_usd.saturating_add(nano);
        }
    }

    pub(crate) fn usd(&self) -> f64 {
        self.nano_usd as f64 / NANO_USD_PER_DOLLAR
    }

    fn input_total(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.cached_input_tokens)
            .saturating_add(self.cache_write_tokens)
    }
}

/// The line appended under a reply. `chat` already includes `reply`.
pub(crate) fn cost_line(reply: &ReplyCost, chat: &ChatCost) -> String {
    let Some(usage) = reply.usage else {
        return "Cost unknown: the provider reported no usage".to_string();
    };
    let Some((usd, source)) = reply.cost else {
        return format!(
            "Tokens {} in · {} out · no price set",
            format_tokens(usage.total_input_tokens()),
            format_tokens(usage.output_tokens),
        );
    };

    let mut line = format!("Cost {}{}", marker(source), format_usd(usd));
    if usage.web_searches > 0 {
        line.push_str(&format!(" ({})", searches(usage.web_searches)));
    }
    line.push_str(&format!(
        " · this chat {}{}",
        chat_marker(chat),
        format_usd(chat.usd())
    ));
    line
}

/// The `/cost` reply. `has_price` says whether the room's model has a price
/// table, which decides the hint for replies whose cost is unknown.
pub(crate) fn summary(
    chat: &ChatCost,
    show_cost: bool,
    model_key: &str,
    has_price: bool,
) -> String {
    let mut lines = Vec::new();
    if chat.replies == 0 {
        lines.push("No replies in this chat yet.".to_string());
    } else {
        let total = if chat.costed_replies == 0 {
            "cost unknown".to_string()
        } else {
            format!("{}{}", chat_marker(chat), format_usd(chat.usd()))
        };
        let basis = if chat.costed_replies == 0 {
            ""
        } else if chat.estimated {
            " (estimated from config.toml prices)"
        } else {
            " (as reported by the provider)"
        };
        lines.push(format!(
            "This chat: {total} over {}{basis}.",
            plural(chat.replies, "reply", "replies")
        ));

        let mut tokens = format!("Tokens: {} in", format_tokens(chat.input_total()));
        let mut input_detail = Vec::new();
        if chat.cached_input_tokens > 0 {
            input_detail.push(format!(
                "{} cached",
                format_tokens(chat.cached_input_tokens)
            ));
        }
        if chat.cache_write_tokens > 0 {
            input_detail.push(format!(
                "{} cache writes",
                format_tokens(chat.cache_write_tokens)
            ));
        }
        if !input_detail.is_empty() {
            tokens.push_str(&format!(" ({})", input_detail.join(", ")));
        }
        tokens.push_str(&format!(" · {} out", format_tokens(chat.output_tokens)));
        if chat.reasoning_tokens > 0 {
            tokens.push_str(&format!(
                " ({} reasoning)",
                format_tokens(chat.reasoning_tokens)
            ));
        }
        if chat.web_searches > 0 {
            tokens.push_str(&format!(" · {}", searches(chat.web_searches)));
        }
        lines.push(tokens);

        let uncosted = chat.replies - chat.costed_replies;
        if uncosted > 0 {
            let reason = if has_price {
                "the provider reported no usage".to_string()
            } else {
                format!("add a [models.{model_key}.price] table to config.toml")
            };
            lines.push(format!(
                "{} had no cost: {reason}.",
                plural(uncosted, "reply", "replies")
            ));
        }
    }
    lines.push(if show_cost {
        "The cost line under each reply is on; /cost off hides it.".to_string()
    } else {
        "The cost line under each reply is off; /cost on shows it.".to_string()
    });
    lines.push("Totals start over with /new.".to_string());
    lines.join("\n")
}

fn marker(source: CostSource) -> &'static str {
    match source {
        CostSource::Reported => "",
        CostSource::Estimated => "≈ ",
    }
}

/// "≥" when some replies have no known cost, so the sum is a lower bound;
/// "≈" when any part of it is an estimate.
fn chat_marker(chat: &ChatCost) -> &'static str {
    if chat.costed_replies < chat.replies {
        "≥ "
    } else if chat.estimated {
        "≈ "
    } else {
        ""
    }
}

fn searches(count: u64) -> String {
    plural(count, "search", "searches")
}

fn plural(count: u64, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}

/// Dollars with enough precision to tell cheap replies apart: four decimals
/// under a cent, three under a dollar, two above.
pub(crate) fn format_usd(usd: f64) -> String {
    if !usd.is_finite() || usd <= 0.0 {
        "$0".to_string()
    } else if usd < 0.0001 {
        "<$0.0001".to_string()
    } else if usd < 0.01 {
        format!("${usd:.4}")
    } else if usd < 1.0 {
        format!("${usd:.3}")
    } else {
        format!("${usd:.2}")
    }
}

pub(crate) fn format_tokens(tokens: u64) -> String {
    let scaled = |value: f64, unit: &str| {
        let text = format!("{value:.1}");
        format!("{}{unit}", text.strip_suffix(".0").unwrap_or(&text))
    };
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 999_950 {
        scaled(tokens as f64 / 1_000.0, "k")
    } else {
        scaled(tokens as f64 / 1_000_000.0, "M")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u64, cached: u64, output: u64, searches: u64) -> Usage {
        Usage {
            input_tokens: input,
            cached_input_tokens: cached,
            cache_write_tokens: 0,
            output_tokens: output,
            reasoning_tokens: 0,
            web_searches: searches,
            reported_cost_usd: None,
        }
    }

    fn price() -> ModelPrice {
        ModelPrice {
            input: 8.0,
            cached_input: Some(0.4),
            cache_write: Some(10.0),
            output: 40.0,
            web_search: Some(10.0),
        }
    }

    #[test]
    fn a_reported_cost_wins_over_the_configured_price() {
        let mut reported = usage(1_000, 0, 1_000, 0);
        reported.reported_cost_usd = Some(0.002);
        let reply = ReplyCost::new(Some(reported), Some(&price()));
        assert_eq!(reply.cost, Some((0.002, CostSource::Reported)));

        let estimated = ReplyCost::new(Some(usage(1_000_000, 0, 0, 0)), Some(&price()));
        assert_eq!(estimated.cost, Some((8.0, CostSource::Estimated)));

        assert_eq!(ReplyCost::new(Some(usage(1, 0, 1, 0)), None).cost, None);
        assert_eq!(ReplyCost::new(None, Some(&price())).cost, None);
    }

    #[test]
    fn cost_line_marks_estimates_searches_and_partial_totals() {
        let mut chat = ChatCost::default();
        let first = ReplyCost::new(Some(usage(10_000, 50_000, 1_000, 2)), Some(&price()));
        chat.record(&first);
        // 0.08 + 0.02 + 0.04 + 0.02
        assert_eq!(
            cost_line(&first, &chat),
            "Cost ≈ $0.160 (2 searches) · this chat ≈ $0.160"
        );

        let unpriced = ReplyCost::new(Some(usage(12_300, 0, 1_240, 0)), None);
        chat.record(&unpriced);
        assert_eq!(
            cost_line(&unpriced, &chat),
            "Tokens 12.3k in · 1.2k out · no price set"
        );

        let mut billed = usage(1_000, 0, 100, 1);
        billed.reported_cost_usd = Some(0.0042);
        let reported = ReplyCost::new(Some(billed), None);
        chat.record(&reported);
        assert_eq!(
            cost_line(&reported, &chat),
            "Cost $0.0042 (1 search) · this chat ≥ $0.164"
        );

        let silent = ReplyCost::new(None, Some(&price()));
        assert_eq!(
            cost_line(&silent, &chat),
            "Cost unknown: the provider reported no usage"
        );
    }

    #[test]
    fn reported_only_totals_carry_no_estimate_marker() {
        let mut chat = ChatCost::default();
        let mut billed = usage(1, 0, 1, 0);
        billed.reported_cost_usd = Some(1.5);
        let reply = ReplyCost::new(Some(billed), None);
        chat.record(&reply);
        chat.record(&reply);
        assert_eq!(cost_line(&reply, &chat), "Cost $1.50 · this chat $3.00");
        assert!(!chat.estimated);
    }

    #[test]
    fn summary_reports_totals_tokens_and_the_line_setting() {
        let empty = summary(&ChatCost::default(), true, "gpt", false);
        assert!(empty.starts_with("No replies in this chat yet."), "{empty}");
        assert!(empty.contains("/cost off hides it"), "{empty}");

        let mut chat = ChatCost::default();
        let mut reply_usage = usage(2_000, 98_000, 9_800, 3);
        reply_usage.cache_write_tokens = 1_000;
        reply_usage.reasoning_tokens = 6_100;
        chat.record(&ReplyCost::new(Some(reply_usage), Some(&price())));
        chat.record(&ReplyCost::new(Some(usage(100, 0, 10, 0)), None));
        let text = summary(&chat, false, "gpt", false);
        assert!(
            text.contains(
                "This chat: ≥ $0.487 over 2 replies (estimated from config.toml prices)."
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "Tokens: 101.1k in (98k cached, 1k cache writes) · 9.8k out (6.1k reasoning) · 3 searches"
            ),
            "{text}"
        );
        assert!(
            text.contains("1 reply had no cost: add a [models.gpt.price] table"),
            "{text}"
        );
        assert!(text.contains("/cost on shows it"), "{text}");

        let priced = summary(&chat, true, "gpt", true);
        assert!(
            priced.contains("1 reply had no cost: the provider reported no usage."),
            "{priced}"
        );
    }

    #[test]
    fn dollars_and_tokens_format_compactly() {
        assert_eq!(format_usd(0.0), "$0");
        assert_eq!(format_usd(0.000_02), "<$0.0001");
        assert_eq!(format_usd(0.004_24), "$0.0042");
        assert_eq!(format_usd(0.0316), "$0.032");
        assert_eq!(format_usd(12.346), "$12.35");
        assert_eq!(format_usd(f64::NAN), "$0");
        assert_eq!(format_tokens(950), "950");
        assert_eq!(format_tokens(1_000), "1k");
        assert_eq!(format_tokens(12_345), "12.3k");
        assert_eq!(format_tokens(999_949), "999.9k");
        assert_eq!(format_tokens(999_950), "1M");
        assert_eq!(format_tokens(2_500_000), "2.5M");
    }
}
