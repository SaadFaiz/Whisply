#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// use std::process::{Command, Child};
// use std::sync::Mutex;
// use std::path::PathBuf;
// use tauri::Manager;
//
// struct PythonProcess(Mutex<Option<Child>>);

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
    //    .manage(PythonProcess(Mutex::new(None)))
    //    .setup(|app| {
    //        let state = app.state::<PythonProcess>();
    //        let mut server_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    //        server_path.pop();
    //        server_path.pop();
    //        server_path.push("server");
    //        server_path.push("main.py");
    //
    //        println!("Looking for Python server at: {:?}", server_path);
    //
    //        let child = Command::new("python")  // Changed from python3 to python
    //            .arg(&server_path)
    //            .spawn()
    //            .expect("Failed to start Python server");
    //
    //     *state.0.lock().unwrap() = Some(child);
    //        Ok(())
    //    })
    //    .on_window_event(|window, event| {
    //        match event {
    //            tauri::WindowEvent::Destroyed => {
    //                if let Some(mut child) = window.state::<PythonProcess>().0.lock().unwrap().take() {
    //                    let _ = child.kill();
    //                }
    //            }
    //            _ => {}
    //        }
    //    })
    //    .run(tauri::generate_context!())
    //    .expect("error while running Tauri application");
}
