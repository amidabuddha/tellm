#![cfg(feature = "keychain")]

use tellm_config::secrets::{self, SecretDestination};

#[test]
#[ignore = "touches the real OS keychain"]
fn real_keychain_roundtrip_probe() {
    let name = format!(
        "tellm_keychain_probe_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    let destination = secrets::set(&name, "probe-value").unwrap();
    assert_eq!(destination, SecretDestination::OsKeychain);
    assert_eq!(secrets::get(&name).as_deref(), Some("probe-value"));

    let entry = keyring_core::Entry::new("tellm", &name).unwrap();
    let _ = entry.delete_credential();
}
