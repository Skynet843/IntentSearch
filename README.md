# 🧠 IntentSearch

**IntentSearch**  is an AI-powered, multimodal intent recognition system built to enhance e-commerce search experiences. Interpreting user inputs—whether text, image, or voice—transforms them into structured queries to enable smarter product searches and personalized recommendations.

## ✨ Features

- **🔍 Multimodal Input Handling**: Accepts and processes text, image, and voice inputs to understand user intent.
- **🧠 Intent Recognition**: Uses NLP and AI models to interpret user intentions and context with high accuracy.
- **🧾 Structured Query Generation**: Transforms natural language inputs into structured queries with fields like `query`, `category`, `price_max`, and `intent`.
- **🖼️ Image Captioning**: Utilizes BLIP (Bootstrapping Language-Image Pretraining) to generate descriptive captions for images, aiding in intent recognition.
- **🔊 Audio Transcription**: Converts voice inputs into text for further processing.
- **🗃️ MongoDB Integration**: Stores and retrieves product data efficiently using MongoDB.
- **⚡ FastAPI Backend**: Provides a robust and scalable API for handling requests.

## 📸 Screenshots or Demo

Replace with actual screenshots or links to demo videos once available.

### 🔍 Multimodal Search Interface

![Multimodal Search UI](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfxUWbpRJO914twhna3q1m49jCKVqRW5SgxDkyXb4yUf-konmdYNANSmbBoRp6D2yM2Io&usqp=CAU)

## 🧰 Tech Stack

- **Backend**: FastAPI
- **Database**: MongoDB
- **AI/NLP**: OpenAI API, BLIP (Bootstrapped Language-Image Pretraining)
- **Audio Processing**: SpeechRecognition
- **Language**: Python 3.8+
- **Deployment**: Uvicorn

## 🚀 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Skynet843/IntentSearch.git
   cd IntentSearch
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv intent_search_env
   source intent_search_env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add the following:
   ```env
   MONGODB_URI=your_mongodb_connection_string
   MONGODB_DB_NAME=your_database_name
   MONGODB_COLLECTION=your_collection_name
   GOOGLE_API_KEY=your_google_api_key
   ```

## 🥪 Usage

1. **🔥 Run the FastAPI Server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **API Endpoint**:
   - **POST** `/search`: Accepts `text`, `image`, and `voice` inputs.
   
   **Form Data Parameters**:
   | Parameter | Type                | Description              |
   |  :---:    |        :---:        |        :---:             |
   | `text`    | `string` (optional) | Text query input         |
   | `image`   | `file` (optional)   | Upload an image file     |
   | `voice`   | `file` (optional)   | Upload an audio/voice file |

   **Example using `curl`**:
   ```bash
   curl -X POST "http://localhost:8000/search" \
   -F "text=Looking for wireless earbuds" \
   -F "image=@path_to_image.jpg" \
   -F "voice=@path_to_audio.wav"
   ```

## 📂 Project Structure

```
IntentSearch/
├── AudioMaster.py          # Handles audio transcription
├── ImageMaster.py          # Generates captions from images
├── IndexCreaterScript.py   # Script to create search index
├── Indexer.py              # Manages indexing of products
├── QueryParser.py          # Parses and structures user queries
├── db.py                   # MongoDB connection and operations
├── main.py                 # FastAPI application
├── openai_query.py         # Interacts with OpenAI API
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
└── .gitignore              # Specifies files to ignore in Git
```

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact

For any inquiries or support, please open an issue in the repository.

