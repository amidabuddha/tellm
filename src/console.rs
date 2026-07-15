use std::fmt::Arguments;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use log::{Level, LevelFilter, Log, Metadata, Record};

const TARGET_PREFIX: &str = "tellm::";
static LOGGER: ConsoleLogger = ConsoleLogger;

pub fn init() {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(LevelFilter::Debug);
    }
}

struct ConsoleLogger;

impl Log for ConsoleLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= Level::Debug
            && (metadata.target() == "tellm" || metadata.target().starts_with(TARGET_PREFIX))
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let timestamp = local_timestamp(SystemTime::now());
        let component = record
            .target()
            .strip_prefix(TARGET_PREFIX)
            .unwrap_or(record.target());
        let line = format_line(&timestamp, record.level(), component, *record.args());
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(stderr, "{line}");
    }

    fn flush(&self) {
        let _ = std::io::stderr().flush();
    }
}

fn format_line(timestamp: &str, level: Level, component: &str, message: Arguments<'_>) -> String {
    format!("{timestamp} {level:<5} {component:<11} {message}")
}

fn local_timestamp(now: SystemTime) -> String {
    let elapsed = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let Ok(seconds) = libc::time_t::try_from(elapsed.as_secs()) else {
        return format!("unix:{}.{:03}", elapsed.as_secs(), elapsed.subsec_millis());
    };
    let millis = elapsed.subsec_millis();

    local_time_parts(seconds)
        .or_else(|| utc_time_parts(seconds))
        .map(|(parts, offset_seconds)| format_timestamp(parts, millis, offset_seconds))
        .unwrap_or_else(|| format!("unix:{}.{millis:03}", elapsed.as_secs()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TimeParts {
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: i32,
}

impl From<&libc::tm> for TimeParts {
    fn from(value: &libc::tm) -> Self {
        Self {
            year: value.tm_year + 1900,
            month: value.tm_mon + 1,
            day: value.tm_mday,
            hour: value.tm_hour,
            minute: value.tm_min,
            second: value.tm_sec,
        }
    }
}

fn format_timestamp(parts: TimeParts, millis: u32, offset_seconds: i64) -> String {
    let sign = if offset_seconds < 0 { '-' } else { '+' };
    let offset = offset_seconds.unsigned_abs();
    let offset_hours = offset / 3_600;
    let offset_minutes = (offset % 3_600) / 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{millis:03}{sign}{offset_hours:02}:{offset_minutes:02}",
        parts.year, parts.month, parts.day, parts.hour, parts.minute, parts.second
    )
}

#[cfg(unix)]
fn local_time_parts(seconds: libc::time_t) -> Option<(TimeParts, i64)> {
    let mut broken_down = std::mem::MaybeUninit::<libc::tm>::uninit();
    // SAFETY: `seconds` and the output allocation both live for this call;
    // localtime_r initializes the output before returning a non-null pointer.
    let result = unsafe { libc::localtime_r(&seconds, broken_down.as_mut_ptr()) };
    if result.is_null() {
        return None;
    }
    // SAFETY: a non-null localtime_r result means the allocation was initialized.
    let broken_down = unsafe { broken_down.assume_init() };
    Some((TimeParts::from(&broken_down), unix_utc_offset(&broken_down)))
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn unix_utc_offset(broken_down: &libc::tm) -> i64 {
    broken_down.tm_gmtoff
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn unix_utc_offset(_broken_down: &libc::tm) -> i64 {
    0
}

#[cfg(windows)]
fn local_time_parts(seconds: libc::time_t) -> Option<(TimeParts, i64)> {
    let mut broken_down = std::mem::MaybeUninit::<libc::tm>::uninit();
    // SAFETY: both pointers are valid for the duration of the call. A zero
    // return from localtime_s means it initialized the output allocation.
    let result = unsafe { libc::localtime_s(broken_down.as_mut_ptr(), &seconds) };
    if result != 0 {
        return None;
    }
    // SAFETY: localtime_s returned success, so the allocation is initialized.
    let broken_down = unsafe { broken_down.assume_init() };

    let mut standard_bias = 0;
    // SAFETY: the pointer refers to writable storage of the required type.
    if unsafe { libc::get_timezone(&mut standard_bias) } != 0 {
        return None;
    }
    let mut total_bias = standard_bias as i64;
    if broken_down.tm_isdst > 0 {
        let mut daylight_bias = 0;
        // SAFETY: the pointer refers to writable storage of the required type.
        if unsafe { libc::get_dstbias(&mut daylight_bias) } != 0 {
            return None;
        }
        total_bias += daylight_bias as i64;
    }

    Some((TimeParts::from(&broken_down), -total_bias))
}

#[cfg(not(any(unix, windows)))]
fn local_time_parts(_seconds: libc::time_t) -> Option<(TimeParts, i64)> {
    None
}

#[cfg(unix)]
fn utc_time_parts(seconds: libc::time_t) -> Option<(TimeParts, i64)> {
    let mut broken_down = std::mem::MaybeUninit::<libc::tm>::uninit();
    // SAFETY: `seconds` and the output allocation both live for this call;
    // gmtime_r initializes the output before returning a non-null pointer.
    let result = unsafe { libc::gmtime_r(&seconds, broken_down.as_mut_ptr()) };
    if result.is_null() {
        return None;
    }
    // SAFETY: a non-null gmtime_r result means the allocation was initialized.
    let broken_down = unsafe { broken_down.assume_init() };
    Some((TimeParts::from(&broken_down), 0))
}

#[cfg(windows)]
fn utc_time_parts(seconds: libc::time_t) -> Option<(TimeParts, i64)> {
    let mut broken_down = std::mem::MaybeUninit::<libc::tm>::uninit();
    // SAFETY: both pointers are valid for the duration of the call. A zero
    // return from gmtime_s means it initialized the output allocation.
    let result = unsafe { libc::gmtime_s(broken_down.as_mut_ptr(), &seconds) };
    if result != 0 {
        return None;
    }
    // SAFETY: gmtime_s returned success, so the allocation is initialized.
    let broken_down = unsafe { broken_down.assume_init() };
    Some((TimeParts::from(&broken_down), 0))
}

#[cfg(not(any(unix, windows)))]
fn utc_time_parts(_seconds: libc::time_t) -> Option<(TimeParts, i64)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_is_rfc3339_with_milliseconds_and_offset() {
        assert_eq!(
            format_timestamp(
                TimeParts {
                    year: 2026,
                    month: 7,
                    day: 15,
                    hour: 18,
                    minute: 42,
                    second: 3,
                },
                219,
                3 * 60 * 60,
            ),
            "2026-07-15T18:42:03.219+03:00"
        );
        assert_eq!(
            format_timestamp(
                TimeParts {
                    year: 2026,
                    month: 1,
                    day: 2,
                    hour: 3,
                    minute: 4,
                    second: 5,
                },
                6,
                -(5 * 60 * 60 + 30 * 60),
            ),
            "2026-01-02T03:04:05.006-05:30"
        );
    }

    #[test]
    fn line_has_fixed_level_and_component_columns() {
        assert_eq!(
            format_line(
                "2026-07-15T18:42:03.219+03:00",
                Level::Info,
                "tellm",
                format_args!("status=running"),
            ),
            "2026-07-15T18:42:03.219+03:00 INFO  tellm       status=running"
        );
    }
}
