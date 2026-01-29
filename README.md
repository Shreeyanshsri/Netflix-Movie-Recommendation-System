🎬 Netflix Movie Recommendation System

A content-based movie recommendation system that suggests similar movies based on textual features using TF-IDF Vectorization and Cosine Similarity.
The application is built with Python and Streamlit and enhanced with real-time movie posters and plots fetched from the OMDb API.

🚀 Features

🔍 Content-Based Recommendations using TF-IDF & Cosine Similarity

🎞️ Movie Similarity Search based on IMDb Top-1000 dataset

🖼️ Live Movie Posters & Plot Summaries via OMDb API

🎛️ Interactive Filters (Top-N recommendations, minimum IMDb rating)

⚡ Cached Model Loading for faster performance

🎨 Netflix-inspired Dark UI

☁️ Deployable on Streamlit Cloud

🧠 Recommendation Logic

Movie metadata is converted into numerical vectors using TF-IDF

Cosine similarity is calculated between movie vectors

Movies are ranked based on similarity scores

Optional IMDb rating filters are applied

Top-N most similar movies are displayed

🛠 Tech Stack

Programming Language: Python

Web Framework: Streamlit

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn

Similarity Handling: SciPy (sparse matrices)

API Integration: OMDb API

Model Storage: Pickle (.pkl)

📂 Project Structure
netflix-movie-recommendation-system/
├── streamlit_app.py                 # Main Streamlit application
├── best_netflix_recommender.pkl     # Trained recommender model
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation

⚙️ Installation & Setup
🔹 Clone the repository
git clone https://github.com/your-username/netflix-movie-recommendation-system.git
cd netflix-movie-recommendation-system

🔹 Install dependencies
pip install -r requirements.txt

🔹 Run the application
streamlit run streamlit_app.py

🌐 Deployment Notes
✔ Local / College Demo

Python: 3.10 or 3.11

Streamlit: 1.19.0

✔ Streamlit Cloud

Python 3.11 supported

Uses pinned dependency versions for compatibility

📊 Dataset

Source: IMDb Top-1000 Movies dataset

Key Attributes:

Movie Title

Genre

IMDb Rating

Director

Overview (used for TF-IDF)

📈 Future Enhancements

🔮 Hybrid recommendation system (content + collaborative)

👤 User profiles & watchlists

🧠 Personalized recommendations

📱 Mobile-responsive UI

📊 Recommendation performance analytics

👨‍🎓 Academic Relevance

This project is suitable for:

BCA / MCA / B.Tech final-year projects

Machine Learning & Data Science portfolios

Demonstrating real-world recommender systems

Mini-projects & viva presentations

🧑‍💻 Author

Shreeyansh Srivastava

📜 License

This project is intended for educational and learning purposes.
You are free to modify and extend it for personal or academic use.
