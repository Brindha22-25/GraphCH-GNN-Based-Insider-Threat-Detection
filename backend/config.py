import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if available)
load_dotenv()

class Config:
    """Base configuration settings."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_default_secret_key')
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///default.db')
    
    # Paths
    MODEL_PATH = os.getenv('MODEL_PATH', '../models/insider_threat_model.pkl')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '../data/uploads')
    
    # File settings
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

class DevelopmentConfig(Config):
    """Configuration for development environment."""
    DEBUG = True
    DATABASE_URI = os.getenv('DEV_DATABASE_URI', 'sqlite:///dev.db')

class TestingConfig(Config):
    """Configuration for testing environment."""
    TESTING = True
    DATABASE_URI = os.getenv('TEST_DATABASE_URI', 'sqlite:///test.db')

class ProductionConfig(Config):
    """Configuration for production deployment."""
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///prod.db')

# Dictionary to access configurations based on environment
configurations = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig
}
