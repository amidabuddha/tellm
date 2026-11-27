use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;

use tellm_config::Config;
use tokio::sync::{Mutex, oneshot};

use crate::rooms::{self, RoomSettings, RoomState, RoomStates};

/// Coordinates runtime-owned config and room-setting persistence.
///
/// The blocking filesystem work stays on one ordered thread. Room mutations
/// additionally share a transaction lock so their in-memory snapshots and
/// rollback behavior cannot interleave.
#[derive(Clone)]
pub(crate) struct Persistence {
    sender: std_mpsc::Sender<PersistenceCommand>,
    room_mutations: Arc<Mutex<()>>,
}

/// One change to config.toml, applied by the writer thread to the file as
/// it is on disk at that moment.
pub(crate) type ConfigChange = Box<dyn FnOnce(&mut Config) + Send>;

enum PersistenceCommand {
    UpdateConfig {
        change: ConfigChange,
        reply: oneshot::Sender<Result<(), String>>,
    },
    SaveRooms {
        settings: BTreeMap<i64, RoomSettings>,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

impl Persistence {
    /// Applies `change` to config.toml as it is on disk now and writes the
    /// result. The runtime's own copy is loaded once at startup, so writing
    /// that copy would revert every edit the owner made to the file since;
    /// applying just the change keeps them.
    pub(crate) async fn update_config(
        &self,
        change: impl FnOnce(&mut Config) + Send + 'static,
    ) -> Result<(), String> {
        let (reply, result) = oneshot::channel();
        self.sender
            .send(PersistenceCommand::UpdateConfig {
                change: Box::new(change),
                reply,
            })
            .map_err(|_| "config persistence writer stopped".to_string())?;
        result
            .await
            .map_err(|_| "config persistence writer stopped before replying".to_string())?
    }

    pub(crate) async fn mutate_room(
        &self,
        rooms: &Mutex<RoomStates>,
        chat_id: i64,
        mutate: impl FnOnce(&mut RoomState) + Send,
    ) -> Result<(), String> {
        let _transaction = self.room_mutations.lock().await;
        let (before, settings) = {
            let mut rooms = rooms.lock().await;
            let before = rooms.get(chat_id).map(|room| room.settings.clone());
            let room = rooms.get_or_default(chat_id);
            mutate(room);
            (before, rooms.settings())
        };
        if let Err(error) = self.save_rooms(settings).await {
            let mut rooms = rooms.lock().await;
            match before {
                Some(before) => {
                    if let Some(current) = rooms.get_mut(chat_id) {
                        // Only settings are durable. Roll them back, but never
                        // resurrect history invalidated by this mutation or by
                        // a concurrent terminal reset.
                        current.settings = before;
                    }
                }
                None => {
                    rooms.remove(chat_id);
                }
            }
            return Err(format!(
                "failed to persist room {chat_id}; settings mutation rolled back: {error}"
            ));
        }
        Ok(())
    }

    /// Removes a denied room and persists the new snapshot.
    ///
    /// Failure deliberately does not restore the in-memory room: access
    /// revocation must not recreate denied conversation state. A later room
    /// mutation retries the complete settings snapshot.
    pub(crate) async fn remove_room(
        &self,
        rooms: &Mutex<RoomStates>,
        chat_id: i64,
    ) -> Result<(), String> {
        let _transaction = self.room_mutations.lock().await;
        let settings = {
            let mut rooms = rooms.lock().await;
            rooms.remove(chat_id);
            rooms.settings()
        };
        self.save_rooms(settings).await
    }

    async fn save_rooms(&self, settings: BTreeMap<i64, RoomSettings>) -> Result<(), String> {
        let (reply, result) = oneshot::channel();
        self.sender
            .send(PersistenceCommand::SaveRooms { settings, reply })
            .map_err(|_| "room persistence writer stopped".to_string())?;
        result
            .await
            .map_err(|_| "room persistence writer stopped before replying".to_string())?
    }

    pub(crate) async fn shutdown(&self) -> Result<(), String> {
        let (reply, done) = oneshot::channel();
        self.sender
            .send(PersistenceCommand::Shutdown { reply })
            .map_err(|_| "persistence writer already stopped".to_string())?;
        done.await
            .map_err(|_| "persistence writer stopped before shutdown completed".to_string())
    }
}

pub(crate) fn spawn() -> std::io::Result<(Persistence, std::thread::JoinHandle<()>)> {
    spawn_with_impl(update_config_file, |settings| {
        rooms::save_settings(&settings).map_err(|error| error.to_string())
    })
}

/// Reads config.toml, applies `change`, and writes it back. A file that no
/// longer parses is left alone: the change fails instead of overwriting
/// whatever the owner is in the middle of editing.
fn update_config_file(change: ConfigChange) -> Result<(), String> {
    let mut config = tellm_config::load().map_err(|error| {
        format!("config.toml could not be read, so it was left unchanged: {error}")
    })?;
    change(&mut config);
    tellm_config::save(&config).map_err(|error| error.to_string())
}

#[cfg(test)]
pub(crate) fn spawn_with<UpdateConfig, SaveRooms>(
    update_config: UpdateConfig,
    save_rooms: SaveRooms,
) -> std::io::Result<(Persistence, std::thread::JoinHandle<()>)>
where
    UpdateConfig: Fn(ConfigChange) -> Result<(), String> + Send + 'static,
    SaveRooms: Fn(BTreeMap<i64, RoomSettings>) -> Result<(), String> + Send + 'static,
{
    spawn_with_impl(update_config, save_rooms)
}

fn spawn_with_impl<UpdateConfig, SaveRooms>(
    update_config: UpdateConfig,
    save_rooms: SaveRooms,
) -> std::io::Result<(Persistence, std::thread::JoinHandle<()>)>
where
    UpdateConfig: Fn(ConfigChange) -> Result<(), String> + Send + 'static,
    SaveRooms: Fn(BTreeMap<i64, RoomSettings>) -> Result<(), String> + Send + 'static,
{
    let (sender, receiver) = std_mpsc::channel();
    let thread = std::thread::Builder::new()
        .name("tellm-persistence".to_string())
        .spawn(move || {
            while let Ok(command) = receiver.recv() {
                match command {
                    PersistenceCommand::UpdateConfig { change, reply } => {
                        let _ = reply.send(update_config(change));
                    }
                    PersistenceCommand::SaveRooms { settings, reply } => {
                        let _ = reply.send(save_rooms(settings));
                    }
                    PersistenceCommand::Shutdown { reply } => {
                        let _ = reply.send(());
                        break;
                    }
                }
            }
        })?;
    Ok((
        Persistence {
            sender,
            room_mutations: Arc::new(Mutex::new(())),
        },
        thread,
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::{Barrier, Mutex as StdMutex};

    use tellm_config::WireFormat;
    use tokio::task::spawn_blocking;

    use super::*;

    #[tokio::test]
    async fn writer_orders_an_abandoned_write_before_the_next_snapshot() {
        let observed = Arc::new(StdMutex::new(Vec::<Vec<i64>>::new()));
        let thread_observed = Arc::clone(&observed);
        let release_first = Arc::new(Barrier::new(2));
        let thread_release = Arc::clone(&release_first);
        let (started_tx, started_rx) = std_mpsc::channel();
        let (persistence, thread) = spawn_with(
            |_change| Ok(()),
            move |settings| {
                let ids = settings.keys().copied().collect::<Vec<_>>();
                if ids == [1] {
                    let _ = started_tx.send(());
                    thread_release.wait();
                }
                thread_observed
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(ids);
                Ok(())
            },
        )
        .expect("spawn persistence writer");

        let first_persistence = persistence.clone();
        let first = tokio::spawn(async move {
            first_persistence
                .save_rooms(BTreeMap::from([(1, RoomSettings::default())]))
                .await
        });
        spawn_blocking(move || started_rx.recv())
            .await
            .expect("join start wait")
            .expect("first write should start");
        first.abort();

        let second_persistence = persistence.clone();
        let second = tokio::spawn(async move {
            second_persistence
                .save_rooms(BTreeMap::from([(2, RoomSettings::default())]))
                .await
        });
        spawn_blocking(move || release_first.wait())
            .await
            .expect("release first write");

        second
            .await
            .expect("second caller should run")
            .expect("second write should succeed");
        shutdown(persistence, thread).await;

        assert_eq!(
            *observed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![1], vec![2]]
        );
    }

    #[tokio::test]
    async fn failed_room_mutation_restores_settings_without_resurrecting_history() {
        let (persistence, thread) = spawn_with(
            |_change| Ok(()),
            |_settings| Err("injected room failure".to_string()),
        )
        .expect("spawn persistence writer");
        let rooms = Mutex::new(RoomStates::default());
        {
            let mut rooms = rooms.lock().await;
            let room = rooms.get_or_default(42);
            room.settings.role = Some("original".to_string());
            room.append_turn(WireFormat::Responses, vec![serde_json::json!({"turn": 1})]);
        }

        let error = persistence
            .mutate_room(&rooms, 42, |room| {
                room.settings.role = Some("changed".to_string());
                room.reset_history();
            })
            .await
            .expect_err("injected save failure should be returned");

        assert!(error.contains("settings mutation rolled back"), "{error}");
        let rooms = rooms.lock().await;
        let room = rooms.get(42).expect("existing room should remain");
        assert_eq!(room.settings.role.as_deref(), Some("original"));
        assert!(room.history.is_empty());
        drop(rooms);
        shutdown(persistence, thread).await;
    }

    #[tokio::test]
    async fn failed_denied_room_cleanup_does_not_restore_the_room() {
        let (persistence, thread) = spawn_with(
            |_change| Ok(()),
            |_settings| Err("injected room failure".to_string()),
        )
        .expect("spawn persistence writer");
        let rooms = Mutex::new(RoomStates::default());
        {
            let mut rooms = rooms.lock().await;
            rooms.get_or_default(42).settings.role = Some("private".to_string());
        }

        persistence
            .remove_room(&rooms, 42)
            .await
            .expect_err("injected save failure should be returned");

        assert!(rooms.lock().await.get(42).is_none());
        shutdown(persistence, thread).await;
    }

    async fn shutdown(persistence: Persistence, thread: std::thread::JoinHandle<()>) {
        persistence
            .shutdown()
            .await
            .expect("writer should shut down");
        spawn_blocking(move || thread.join())
            .await
            .expect("join persistence thread")
            .expect("persistence thread should not panic");
    }
}
