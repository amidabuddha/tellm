use std::process::ExitCode;

mod access;
mod commands;
mod rooms;
mod runtime;
mod wizard;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("tellm: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        return run_subcommand(&args);
    }

    let config_path = tellm_config::config_path()?;
    if !config_path.exists() {
        let stdin = std::io::stdin();
        let mut input = stdin.lock();
        let mut output = std::io::stdout();
        wizard::run_first_run(&mut input, &mut output).await?;
    }

    let config = tellm_config::load_validated()?;
    runtime::Runtime::new(config)?.run().await
}

/// `tellm secret set NAME` — store a provider secret from the console with
/// hidden input. Keys must never travel through Telegram chat; this
/// is the supported way to add or rotate them. Secrets are read per request,
/// so a running bot picks the new value up without a restart.
fn run_subcommand(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    match args {
        [command, action, name] if command == "secret" && action == "set" => {
            let value = rpassword::prompt_password(format!("Value for {name} (hidden): "))?;
            let Some(destination) = tellm_config::secrets::set_nonempty(name, &value)? else {
                return Err("empty secret value".into());
            };
            println!("{name} stored in {}.", destination.location_label());
            Ok(())
        }
        _ => Err(format!(
            "unknown arguments: {}. Usage: tellm [secret set NAME]",
            args.join(" ")
        )
        .into()),
    }
}
