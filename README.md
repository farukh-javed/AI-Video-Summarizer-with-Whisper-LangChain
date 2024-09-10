# 🎬 YouTube Video Summarizer

## Overview

This Streamlit app allows you to:
1. Download a YouTube video from a provided URL.
2. Display the video's thumbnail, title, and author.
3. Transcribe the video's audio.
4. Summarize the video's content in bullet points.
5. Answer user questions based on the video's content.

## 🚀 Getting Started

### Prerequisites

- **Python**: Ensure you have Python installed on your machine.
- **ffmpeg**: Required for video processing. Follow the instructions below to install it.

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/farukh-javed/AI-Video-Summarizer-with-Whisper-LangChain.git
    cd AI-Video-Summarizer-with-Whisper-LangChain
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 🔧 Setting Up `ffmpeg`

`ffmpeg` is required for video processing. Here's how to install it for different environments:

#### **Windows**

1. Download the latest version of `ffmpeg` from the [FFmpeg website](https://ffmpeg.org/download.html).
2. Extract the downloaded ZIP file.
3. Add the `bin` directory from the extracted files to your system's PATH:
   - Right-click on `This PC` or `My Computer` and select `Properties`.
   - Click on `Advanced system settings`.
   - Click on `Environment Variables`.
   - Find the `Path` variable in the `System variables` section and click `Edit`.
   - Click `New` and add the path to the `bin` directory.
   - Click `OK` to save and close all dialog boxes.

#### **macOS**

1. Install `ffmpeg` using Homebrew:
    ```bash
    brew install ffmpeg
    ```

#### **Linux**

1. Install `ffmpeg` using your package manager. For example, on Debian-based systems:
    ```bash
    sudo apt-get update
    sudo apt-get install ffmpeg
    ```

### 🛠️ Running the App

1. **Set up environment variables**:
   Create a `.env` file in the project directory with the following content:
   ```
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

2. **Run the app**:
    ```bash
    streamlit run app.py
    ```

### Usage

1. Enter a YouTube video URL in the input field.
2. Click the **"Process Video"** button.
3. View the video's thumbnail, title, and author.
4. Read the summarized points of the video.
5. Ask questions about the video to get detailed answers.

### 🤖 How It Works

1. **Download Video**: Uses `yt_dlp` to download the video in MP4 format.
2. **Transcribe Audio**: Uses the Groq API to transcribe the video's audio.
3. **Text Splitting**: Splits the transcribed text into smaller chunks for processing.
4. **Create Embeddings**: Uses DeepLake to create embeddings of the text.
5. **Retrieve Responses**: Uses LangChain to retrieve summarized responses and answer questions.

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.