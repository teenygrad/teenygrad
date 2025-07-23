/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use teeny_http::download::download_file;
use teeny_http::fetch::fetch_content;
use tokio::fs::create_dir_all;
use tracing::info;

use crate::error::{Error, Result};

/// Represents a file in a Hugging Face model repository
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct HfFile {
    #[serde(rename = "type")]
    file_type: String,
    size: Option<u64>,
    oid: String,
    path: String,
    #[serde(rename = "xetHash")]
    xet_hash: Option<String>,

    #[serde(rename = "lfs")]
    lfs_info: Option<LfsInfo>,
}

/// LFS (Large File Storage) information for large files
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct LfsInfo {
    oid: String,
    #[serde(rename = "pointerSize")]
    pointer_size: u64,
    size: u64,
}

/// Represents the response from the Hugging Face API when listing files
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct HfApiResponse {
    #[serde(rename = "type")]
    response_type: String,
    entries: Vec<HfFile>,
}

/// Configuration for downloading a Hugging Face model
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// The model identifier (e.g., "bert-base-uncased")
    pub model_id: String,

    /// The revision/branch to download (default: "main")
    pub revision: String,

    /// Directory to save the model files
    pub output_dir: PathBuf,

    /// Whether to include tokenizer files
    pub include_tokenizer: bool,

    /// Whether to include configuration files
    pub include_config: bool,

    /// Whether to include model weights
    pub include_weights: bool,

    /// Optional authentication token for private models
    pub auth_token: Option<String>,

    /// Maximum concurrent downloads
    pub max_concurrent: usize,

    /// Whether to show progress
    pub show_progress: bool,
}

/// Downloads a Hugging Face model to the specified directory
pub async fn download_model(config: DownloadConfig) -> Result<()> {
    // Validate model ID
    if config.model_id.is_empty() {
        return Err(Error::InvalidModelId {
            model_id: config.model_id.clone(),
        });
    }

    let output_dir = config.output_dir.join(&config.model_id);

    // Create output directory
    create_dir_all(&output_dir)
        .await
        .map_err(|e| Error::IoError(io::Error::other(e)))?;

    // Build API URL for listing files
    let api_url = format!(
        "https://huggingface.co/api/models/{}/tree/{}",
        config.model_id, config.revision
    );

    // Prepare headers
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("User-Agent", "teeny-http/1.0".parse()?);

    if let Some(token) = &config.auth_token {
        headers.insert("Authorization", format!("Bearer {token}").parse()?);
    }

    // Fetch file list from Hugging Face API
    // info!("Fetching file list for model: {}", config.model_id);
    // let response = client
    //     .get(&api_url)
    //     .headers(headers.clone())
    //     .send()
    //     .await
    //     .map_err(Error::TeenyHttpError)?;

    // info!("Response: {:?}", response);

    let content = fetch_content(&config.model_id, &api_url, Some(headers.clone()), true)
        .await
        .map_err(Error::TeenyHttpError)?;
    let entries: Vec<HfFile> = serde_json::from_slice(&content)?;

    // Filter files based on configuration
    let files_to_download: Vec<HfFile> = entries
        .into_iter()
        .filter(|file| {
            let name = file.path.split("/").last().unwrap();
            let path = output_dir.join(name);

            if path.exists() {
                return false;
            }

            true
        })
        .filter(|file| {
            let path = &file.path;

            // Include tokenizer files
            if config.include_tokenizer && is_tokenizer_file(path).is_ok() {
                return true;
            }

            // Include configuration files
            if config.include_config && is_config_file(path).is_ok() {
                return true;
            }

            // Include model weight files
            if config.include_weights && is_weight_file(path).is_ok() {
                return true;
            }

            false
        })
        .collect();

    if files_to_download.is_empty() {
        info!("No files found to download for the specified configuration");
        return Ok(());
    }

    for (i, file) in files_to_download.iter().enumerate() {
        let headers = headers.clone();
        let name = file.path.split("/").last().unwrap();

        // Determine the download URL
        let download_url = if let Some(_lfs_info) = &file.lfs_info {
            // For LFS files, we need to get the actual download URL
            format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                config.model_id, config.revision, &file.path
            )
        } else {
            // For regular files
            format!(
                "https://huggingface.co/{}/raw/{}/{}",
                config.model_id, config.revision, &file.path
            )
        };

        println!(
            "Downloading file: {}/{}: {name} from {download_url}",
            i + 1,
            files_to_download.len()
        );

        download_file(name, &download_url, &output_dir, Some(headers), true).await?;
    }

    Ok(())
}

// /// Downloads a single file from Hugging Face with progress tracking
// async fn download_file_with_progress(
//     client: &Client,
//     headers: &reqwest::header::HeaderMap,
//     config: &DownloadConfig,
//     file: &HfFile,
//     progress: &Arc<DownloadProgress>,
//     completed_files: &Arc<AtomicU64>,
//     downloaded_bytes: &Arc<AtomicU64>,
// ) -> Result<()> {
//     let file_path = &file.path;

//     // Update current file in progress
//     {
//         let mut current_progress = progress.as_ref().clone();
//         current_progress.current_file = Some(file_path.clone());
//         current_progress.current_file_progress = 0.0;

//         if let Some(callback) = &config.progress_callback {
//             callback(&current_progress);
//         }
//     }

//     // Determine the download URL
//     let download_url = if let Some(_lfs_info) = &file.lfs_info {
//         // For LFS files, we need to get the actual download URL
//         format!(
//             "https://huggingface.co/{}/resolve/{}/{}",
//             config.model_id, config.revision, file_path
//         )
//     } else {
//         // For regular files
//         format!(
//             "https://huggingface.co/{}/raw/{}/{}",
//             config.model_id, config.revision, file_path
//         )
//     };

//     // Create the local file path
//     let local_path = config
//         .output_dir
//         .join(format!("{}/{}", config.model_id, file_path));

//     // Create parent directories if they don't exist
//     if let Some(parent) = local_path.parent() {
//         create_dir_all(parent)
//             .await
//             .map_err(|e| Error::IoError(io::Error::other(e)))?;
//     }

//     // Check if file already exists and has the correct size
//     if local_path.exists() {
//         if let Some(expected_size) = file.size {
//             if let Ok(metadata) = std::fs::metadata(&local_path) {
//                 if metadata.len() == expected_size {
//                     info!("File already exists and size matches: {}", file_path);
//                     return Ok(());
//                 }
//             }
//         }
//     }

//     info!("Downloading: {}", file_path);

//     // Download the file
//     let response = client
//         .get(&download_url)
//         .headers(headers.clone())
//         .send()
//         .await
//         .map_err(Error::TeenyHttpError)?;

//     if !response.status().is_success() {
//         return Err(Error::DownloadFailed {
//             file_path: file_path.clone(),
//             reason: format!("HTTP {}", response.status()),
//         });
//     }

//     // Get file size for progress tracking
//     let content_length = response.content_length();

//     // Download and save the file
//     let bytes = response.bytes().await.map_err(Error::TeenyHttpError)?;

//     // Write to file
//     let mut file = File::create(&local_path).map_err(Error::IoError)?;
//     file.write_all(&bytes).map_err(Error::IoError)?;

//     if let Some(expected_size) = content_length {
//         if bytes.len() as u64 != expected_size {
//             return Err(Error::DownloadFailed {
//                 file_path: file_path.clone(),
//                 reason: format!(
//                     "Size mismatch: expected {}, got {}",
//                     expected_size,
//                     bytes.len()
//                 ),
//             });
//         }
//     }

//     // Update progress
//     let completed = completed_files.fetch_add(1, Ordering::Relaxed) + 1;
//     let downloaded =
//         downloaded_bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed) + bytes.len() as u64;
//     let mut current_progress = progress.as_ref().clone();
//     current_progress.completed_files = completed as usize;
//     current_progress.downloaded_bytes = downloaded;
//     current_progress.current_file = None;
//     current_progress.current_file_progress = 1.0;

//     if let Some(callback) = &config.progress_callback {
//         callback(&current_progress);
//     }

//     info!("Downloaded: {} ({} bytes)", file_path, bytes.len());
//     Ok(())
// }

/// Checks if a file is a tokenizer file
fn is_tokenizer_file(path: &str) -> Result<bool> {
    let tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
        "tokenizer.model",
        "bpe.codes",
        "bpe.vocab",
    ];

    let filename = Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| Error::FileNotFound {
            file_path: path.to_string(),
        })?;

    Ok(tokenizer_files.contains(&filename))
}

/// Checks if a file is a configuration file
fn is_config_file(path: &str) -> Result<bool> {
    let config_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "feature_extractor_config.json",
        "processor_config.json",
    ];

    let filename = Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| Error::FileNotFound {
            file_path: path.to_string(),
        })?;

    Ok(config_files.contains(&filename))
}

/// Checks if a file is a model weight file
fn is_weight_file(path: &str) -> Result<bool> {
    let weight_extensions = [
        ".bin",
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
        ".h5",
        ".pb",
        ".onnx",
    ];

    let path_lower = path.to_lowercase();
    Ok(weight_extensions
        .iter()
        .any(|ext| path_lower.ends_with(ext)))
}

// Downloads a specific file from a Hugging Face model
// pub async fn download_specific_file(
//     model_id: &str,
//     file_path: &str,
//     output_path: &Path,
//     auth_token: Option<&str>,
// ) -> Result<()> {
//     let _config = DownloadConfig {
//         model_id: model_id.to_string(),
//         revision: "main".to_string(),
//         output_dir: output_path.parent().unwrap_or(Path::new(".")).to_path_buf(),
//         include_tokenizer: false,
//         include_config: false,
//         include_weights: false,
//         auth_token: auth_token.map(|s| s.to_string()),
//         max_concurrent: 1,
//         progress_callback: None,
//     };

//     let client = Client::builder().build().map_err(Error::TeenyHttpError)?;

//     let mut headers = reqwest::header::HeaderMap::new();
//     headers.insert(
//         "User-Agent",
//         "rust-huggingface-downloader/1.0"
//             .parse()
//             .map_err(Error::InvalidHeaderValue)?,
//     );

//     if let Some(token) = auth_token {
//         headers.insert(
//             "Authorization",
//             format!("Bearer {token}")
//                 .parse()
//                 .map_err(Error::InvalidHeaderValue)?,
//         );
//     }

//     let download_url = format!("https://huggingface.co/{model_id}/resolve/main/{file_path}");

//     let response = client
//         .get(&download_url)
//         .headers(headers)
//         .send()
//         .await
//         .map_err(Error::TeenyHttpError)?;

//     if !response.status().is_success() {
//         return Err(Error::DownloadFailed {
//             file_path: file_path.to_string(),
//             reason: format!("HTTP {}", response.status()),
//         });
//     }

//     let bytes = response.bytes().await.map_err(Error::TeenyHttpError)?;

//     // Create parent directories if they don't exist
//     if let Some(parent) = output_path.parent() {
//         create_dir_all(parent)
//             .await
//             .map_err(|e| Error::IoError(io::Error::other(e)))?;
//     }

//     let mut file = File::create(output_path).map_err(Error::IoError)?;
//     file.write_all(&bytes).map_err(Error::IoError)?;

//     info!("Downloaded: {} to {}", file_path, output_path.display());
//     Ok(())
// }

// Downloads a single file from Hugging Face (without progress tracking)
// async fn _download_file(
//     client: &Client,
//     headers: &reqwest::header::HeaderMap,
//     config: &DownloadConfig,
//     file: &HfFile,
// ) -> Result<()> {
//     let file_path = &file.path;

//     // Determine the download URL
//     let download_url = if let Some(_lfs_info) = &file.lfs_info {
//         // For LFS files, we need to get the actual download URL
//         format!(
//             "https://huggingface.co/{}/resolve/{}/{}",
//             config.model_id, config.revision, file_path
//         )
//     } else {
//         // For regular files
//         format!(
//             "https://huggingface.co/{}/raw/{}/{}",
//             config.model_id, config.revision, file_path
//         )
//     };

//     // Create the local file path
//     let local_path = config
//         .output_dir
//         .join(format!("{}/{}", config.model_id, file_path));

//     // Create parent directories if they don't exist
//     if let Some(parent) = local_path.parent() {
//         create_dir_all(parent)
//             .await
//             .map_err(|e| Error::IoError(io::Error::other(e)))?;
//     }

//     // Check if file already exists and has the correct size
//     if local_path.exists() {
//         if let Some(expected_size) = file.size {
//             if let Ok(metadata) = std::fs::metadata(&local_path) {
//                 if metadata.len() == expected_size {
//                     info!("File already exists and size matches: {}", file_path);
//                     return Ok(());
//                 }
//             }
//         }
//     }

//     info!("Downloading: {}", file_path);

//     // Download the file
//     let response = client
//         .get(&download_url)
//         .headers(headers.clone())
//         .send()
//         .await
//         .map_err(Error::TeenyHttpError)?;

//     if !response.status().is_success() {
//         return Err(Error::DownloadFailed {
//             file_path: file_path.clone(),
//             reason: format!("HTTP {}", response.status()),
//         });
//     }

//     // Get file size for progress tracking
//     let content_length = response.content_length();

//     // Download and save the file
//     let bytes = response.bytes().await.map_err(Error::TeenyHttpError)?;

//     // Write to file
//     let mut file = File::create(&local_path).map_err(Error::IoError)?;
//     file.write_all(&bytes).map_err(Error::IoError)?;

//     if let Some(expected_size) = content_length {
//         if bytes.len() as u64 != expected_size {
//             return Err(Error::DownloadFailed {
//                 file_path: file_path.clone(),
//                 reason: format!(
//                     "Size mismatch: expected {}, got {}",
//                     expected_size,
//                     bytes.len()
//                 ),
//             });
//         }
//     }

//     info!("Downloaded: {} ({} bytes)", file_path, bytes.len());
//     Ok(())
// }

// Example usage of the Hugging Face downloader with progress tracking
// pub async fn example_usage_with_progress() -> Result<()> {
//     // Example: Download a complete model with progress tracking
//     let config = DownloadConfig {
//         model_id: "bert-base-uncased".to_string(),
//         revision: "main".to_string(),
//         output_dir: PathBuf::from("./downloaded_bert"),
//         include_tokenizer: true,
//         include_config: true,
//         include_weights: true,
//         auth_token: None, // Set this if downloading private models
//         max_concurrent: 4,
//         progress_callback: Some(default_progress_callback()),
//     };

//     info!("Starting download with progress tracking...");
//     download_model(config).await?;
//     info!("Download completed successfully!");

//     Ok(())
// }

// pub async fn example_usage() -> Result<()> {
