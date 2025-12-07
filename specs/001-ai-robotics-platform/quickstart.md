# Quickstart Guide: Physical AI & Humanoid Robotics Textbook Platform

**Feature Branch**: `001-ai-robotics-platform` | **Date**: 2025-12-05

This guide provides instructions to quickly set up and run the Physical AI & Humanoid Robotics Textbook Platform locally.

## 1. Prerequisites

Ensure you have the following installed:

-   **Git**: For cloning the repository.
-   **Python 3.11+**: For the FastAPI backend.
-   **Node.js (LTS)**: For the Docusaurus frontend.
-   **npm** or **yarn**: Node.js package manager.
-   **Docker (Optional)**: If you prefer to run databases locally via Docker.
-   **Qdrant Cloud Account (Free Tier)**: For the vector database.
-   **Neon Serverless Postgres Account (Free Tier)**: For the relational database.

## 2. Clone the Repository

```bash
git clone [REPOSITORY_URL]
cd hacbook # Or your repository name
```

## 3. Backend Setup (FastAPI)

1.  **Navigate to the backend directory**:
    ```bash
    cd backend
    ```

2.  **Create a Python virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt # (File to be created later)
    ```

4.  **Configure Environment Variables**:
    Create a `.env` file in the `backend/` directory with the following (fill in your actual values):
    ```ini
    DATABASE_URL="postgresql://user:password@host:port/database"
    QDRANT_URL="https://[YOUR_QDRANT_CLUSTER_URL]"
    QDRANT_API_KEY="[YOUR_QDRANT_API_KEY]"
    OPENAI_API_KEY="[YOUR_OPENAI_API_KEY]"
    SECRET_KEY="[YOUR_FASTAPI_SECRET_KEY]"
    ALGORITHM="HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES=30
    ```

5.  **Run database migrations**:
    ```bash
    # Command to run migrations (to be defined, e.g., alembic upgrade head)
    ```

6.  **Start the FastAPI application**:
    ```bash
    uvicorn src.main:app --reload
    ```
    The backend API will be available at `http://localhost:8000`.

## 4. Frontend Setup (Docusaurus)

1.  **Navigate to the frontend directory**:
    ```bash
    cd ../frontend # From backend directory
    ```

2.  **Install Node.js dependencies**:
    ```bash
    npm install # or yarn install
    ```

3.  **Start the Docusaurus development server**:
    ```bash
    npm start # or yarn start
    ```
    The frontend textbook will be available at `http://localhost:3000`.

## 5. Basic Usage

-   **Access the Textbook**: Open your browser to `http://localhost:3000`.
-   **Sign Up**: Navigate to the login/signup page (to be implemented) and create a new account. Fill in your profile details for personalization.
-   **Log In**: Use your registered credentials to log in.
-   **Personalization**: Browse chapters. Use the personalization toggle to switch between difficulty modes (Beginner, Intermediate, Advanced) and language (English, Urdu).
-   **Chatbot Interaction**: Use the embedded chatbot to ask questions about the content. Try selecting text in a chapter and asking a question to test the selected-text answering mode.
