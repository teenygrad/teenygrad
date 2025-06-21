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

use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::Client;
use serde::Deserialize;
use tokio::fs::create_dir_all;

pub mod error;

use error::{DownloadError, DownloadResult};

/// Represents a file in a Hugging Face model repository
#[derive(Debug, Deserialize)]
struct HfFile {
    #[serde(rename = "type")]
    file_type: String,
    size: Option<u64>,
    oid: String,
    path: String,
    xetHash: Option<String>,

    #[serde(rename = "lfs")]
    lfs_info: Option<LfsInfo>,
}

/// LFS (Large File Storage) information for large files
#[derive(Debug, Deserialize)]
struct LfsInfo {
    oid: String,
    #[serde(rename = "pointerSize")]
    pointer_size: u64,
    size: u64,
}

/// Represents the response from the Hugging Face API when listing files
#[derive(Debug, Deserialize)]
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

    /// Timeout for HTTP requests
    pub timeout: Duration,

    /// Maximum concurrent downloads
    pub max_concurrent: usize,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            revision: "main".to_string(),
            output_dir: PathBuf::from("./downloaded_model"),
            include_tokenizer: true,
            include_config: true,
            include_weights: true,
            auth_token: None,
            timeout: Duration::from_secs(300), // 5 minutes
            max_concurrent: 4,
        }
    }
}

/// Downloads a Hugging Face model to the specified directory
pub async fn download_huggingface_model(config: DownloadConfig) -> DownloadResult<()> {
    // Validate model ID
    if config.model_id.is_empty() {
        return Err(DownloadError::InvalidModelId {
            model_id: config.model_id.clone(),
        });
    }

    // Create output directory
    create_dir_all(&config.output_dir)
        .await
        .map_err(|e| DownloadError::IoError(io::Error::other(e)))?;

    // Create HTTP client
    let client = Client::builder()
        .timeout(config.timeout)
        .build()
        .map_err(DownloadError::HttpError)?;

    // Build API URL for listing files
    let api_url = format!(
        "https://huggingface.co/api/models/{}/tree/{}",
        config.model_id, config.revision
    );

    // Prepare headers
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("User-Agent", "rust-huggingface-downloader/1.0".parse()?);

    if let Some(token) = &config.auth_token {
        headers.insert("Authorization", format!("Bearer {}", token).parse()?);
    }

    // Fetch file list from Hugging Face API
    println!("Fetching file list for model: {}", config.model_id);
    let response = client
        .get(&api_url)
        .headers(headers.clone())
        .send()
        .await
        .map_err(DownloadError::HttpError)?;

    println!("Response: {:?}", response);

    if response.status() == 404 {
        return Err(DownloadError::ModelNotFound {
            model_id: config.model_id.clone(),
        });
    }

    if response.status() == 401 {
        return Err(DownloadError::AuthenticationRequired {
            model_id: config.model_id.clone(),
        });
    }

    if !response.status().is_success() {
        return Err(DownloadError::InternalServerError {
            status_code: response.status(),
        });
    }

    let entries: Vec<HfFile> = response.json().await?;
    println!("API Response: {:?}", entries.len());

    // Filter files based on configuration
    let files_to_download: Vec<HfFile> = entries
        .into_iter()
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
        println!("No files found to download for the specified configuration");
        return Ok(());
    }

    println!("Found {} files to download", files_to_download.len());

    // Download files concurrently
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(config.max_concurrent));
    let mut download_tasks = Vec::new();

    for file in files_to_download {
        let semaphore = semaphore.clone();
        let client = client.clone();
        let headers = headers.clone();
        let config = config.clone();

        let task = tokio::spawn(async move {
            let _permit = semaphore.acquire().await?;
            download_file(&client, &headers, &config, &file).await
        });

        download_tasks.push(task);
    }

    // Wait for all downloads to complete
    let mut success_count = 0;
    let mut error_count = 0;

    for task in download_tasks {
        match task.await {
            Ok(Ok(())) => success_count += 1,
            Ok(Err(e)) => {
                error_count += 1;
                eprintln!("Download error: {:?}", e);
            }
            Err(e) => {
                error_count += 1;
                eprintln!("Task error: {:?}", e);
            }
        }
    }

    println!(
        "Download completed: {} successful, {} failed",
        success_count, error_count
    );

    if error_count > 0 {
        return Err(DownloadError::DownloadFailed {
            file_path: "multiple files".to_string(),
            reason: format!("{} files failed to download", error_count),
        });
    }

    Ok(())
}

/// Downloads a single file from Hugging Face
async fn download_file(
    client: &Client,
    headers: &reqwest::header::HeaderMap,
    config: &DownloadConfig,
    file: &HfFile,
) -> DownloadResult<()> {
    let file_path = &file.path;

    // Determine the download URL
    let download_url = if let Some(_lfs_info) = &file.lfs_info {
        // For LFS files, we need to get the actual download URL
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            config.model_id, config.revision, file_path
        )
    } else {
        // For regular files
        format!(
            "https://huggingface.co/{}/raw/{}/{}",
            config.model_id, config.revision, file_path
        )
    };

    // Create the local file path
    let local_path = config.output_dir.join(file_path);

    // Create parent directories if they don't exist
    if let Some(parent) = local_path.parent() {
        create_dir_all(parent)
            .await
            .map_err(|e| DownloadError::IoError(io::Error::other(e)))?;
    }

    println!("Downloading: {}", file_path);

    // Download the file
    let response = client
        .get(&download_url)
        .headers(headers.clone())
        .send()
        .await
        .map_err(DownloadError::HttpError)?;

    if !response.status().is_success() {
        return Err(DownloadError::DownloadFailed {
            file_path: file_path.clone(),
            reason: format!("HTTP {}", response.status()),
        });
    }

    // Get file size for progress tracking
    let content_length = response.content_length();

    // Download and save the file
    let bytes = response.bytes().await.map_err(DownloadError::HttpError)?;

    // Write to file
    let mut file = File::create(&local_path).map_err(DownloadError::IoError)?;
    file.write_all(&bytes).map_err(DownloadError::IoError)?;

    if let Some(expected_size) = content_length {
        if bytes.len() as u64 != expected_size {
            return Err(DownloadError::DownloadFailed {
                file_path: file_path.clone(),
                reason: format!(
                    "Size mismatch: expected {}, got {}",
                    expected_size,
                    bytes.len()
                ),
            });
        }
    }

    println!("Downloaded: {} ({} bytes)", file_path, bytes.len());
    Ok(())
}

/// Checks if a file is a tokenizer file
fn is_tokenizer_file(path: &str) -> Result<bool, DownloadError> {
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
        .ok_or_else(|| DownloadError::FileNotFound {
            file_path: path.to_string(),
        })?;

    Ok(tokenizer_files.contains(&filename))
}

/// Checks if a file is a configuration file
fn is_config_file(path: &str) -> Result<bool, DownloadError> {
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
        .ok_or_else(|| DownloadError::FileNotFound {
            file_path: path.to_string(),
        })?;

    Ok(config_files.contains(&filename))
}

/// Checks if a file is a model weight file
fn is_weight_file(path: &str) -> Result<bool, DownloadError> {
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

/// Downloads a specific file from a Hugging Face model
pub async fn download_specific_file(
    model_id: &str,
    file_path: &str,
    output_path: &Path,
    auth_token: Option<&str>,
) -> DownloadResult<()> {
    let config = DownloadConfig {
        model_id: model_id.to_string(),
        revision: "main".to_string(),
        output_dir: output_path.parent().unwrap_or(Path::new(".")).to_path_buf(),
        include_tokenizer: false,
        include_config: false,
        include_weights: false,
        auth_token: auth_token.map(|s| s.to_string()),
        timeout: Duration::from_secs(300),
        max_concurrent: 1,
    };

    let client = Client::builder()
        .timeout(config.timeout)
        .build()
        .map_err(DownloadError::HttpError)?;

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        "User-Agent",
        "rust-huggingface-downloader/1.0"
            .parse()
            .map_err(DownloadError::InvalidHeaderValue)?,
    );

    if let Some(token) = auth_token {
        headers.insert(
            "Authorization",
            format!("Bearer {}", token)
                .parse()
                .map_err(DownloadError::InvalidHeaderValue)?,
        );
    }

    let download_url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model_id, file_path
    );

    let response = client
        .get(&download_url)
        .headers(headers)
        .send()
        .await
        .map_err(DownloadError::HttpError)?;

    if !response.status().is_success() {
        return Err(DownloadError::DownloadFailed {
            file_path: file_path.to_string(),
            reason: format!("HTTP {}", response.status()),
        });
    }

    let bytes = response.bytes().await.map_err(DownloadError::HttpError)?;

    // Create parent directories if they don't exist
    if let Some(parent) = output_path.parent() {
        create_dir_all(parent)
            .await
            .map_err(|e| DownloadError::IoError(io::Error::other(e)))?;
    }

    let mut file = File::create(output_path).map_err(DownloadError::IoError)?;
    file.write_all(&bytes).map_err(DownloadError::IoError)?;

    println!("Downloaded: {} to {}", file_path, output_path.display());
    Ok(())
}

// pub async fn example_usage() -> DownloadResult<()> {
//     // Example 1: Download a complete model
//     let config = DownloadConfig {
//         model_id: "bert-base-uncased".to_string(),
//         revision: "main".to_string(),
//         output_dir: PathBuf::from("./downloaded_bert"),
//         include_tokenizer: true,
//         include_config: true,
//         include_weights: true,
//         auth_token: None, // Set this if downloading private models
//         timeout: Duration::from_secs(300),
//         max_concurrent: 4,
//     };

//     download_huggingface_model(config).await?;

//     // Example 2: Download a specific file
//     download_specific_file(
//         "bert-base-uncased",
//         "config.json",
//         Path::new("./config.json"),
//         None,
//     )
//     .await?;

//     Ok(())
// }
