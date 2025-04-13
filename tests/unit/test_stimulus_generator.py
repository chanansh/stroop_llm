import os
import shutil
import pytest
from PIL import Image
from src.stimulus_generator import StimulusGenerator
from src.config import ExperimentConfig

@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temporary directory for test images."""
    original_dir = ExperimentConfig.IMAGE_DIR
    ExperimentConfig.IMAGE_DIR = str(tmp_path)
    yield tmp_path
    ExperimentConfig.IMAGE_DIR = original_dir

@pytest.fixture
def stimulus_generator(temp_image_dir):
    """Create a StimulusGenerator instance with temporary directory."""
    return StimulusGenerator()

def test_init_creates_directory(temp_image_dir):
    """Test that __init__ creates the image directory."""
    assert os.path.exists(temp_image_dir)

def test_get_image_filename(stimulus_generator):
    """Test that get_image_filename returns correct path."""
    word = "TEST"
    color = (255, 0, 0)
    expected = os.path.join(ExperimentConfig.IMAGE_DIR, f"{word}_{color[0]}_{color[1]}_{color[2]}.png")
    assert stimulus_generator._get_image_filename(word, color) == expected

def test_create_stimulus_image(stimulus_generator):
    """Test that create_stimulus_image creates a valid image file."""
    word = "TEST"
    color = (255, 0, 0)
    filename = stimulus_generator._get_image_filename(word, color)
    
    # Create the image
    stimulus_generator._create_stimulus_image(word, color)
    
    # Verify file exists
    assert os.path.exists(filename)
    
    # Verify image properties
    with Image.open(filename) as img:
        assert img.size == (ExperimentConfig.IMG_WIDTH, ExperimentConfig.IMG_HEIGHT)
        assert img.mode == "RGB"

def test_get_stimulus_path_creates_image(stimulus_generator):
    """Test that get_stimulus_path creates image if it doesn't exist."""
    word = "TEST"
    color = (255, 0, 0)
    filename = stimulus_generator._get_image_filename(word, color)
    
    # Remove file if it exists
    if os.path.exists(filename):
        os.remove(filename)
    
    # Get path (should create image)
    path = stimulus_generator.get_stimulus_path(word, color)
    
    # Verify file was created
    assert os.path.exists(path)
    assert path == filename

def test_get_stimulus_path_uses_existing(stimulus_generator):
    """Test that get_stimulus_path uses existing image."""
    word = "TEST"
    color = (255, 0, 0)
    filename = stimulus_generator._get_image_filename(word, color)
    
    # Create initial image
    stimulus_generator._create_stimulus_image(word, color)
    initial_mtime = os.path.getmtime(filename)
    
    # Get path (should use existing)
    path = stimulus_generator.get_stimulus_path(word, color)
    
    # Verify file wasn't recreated
    assert os.path.getmtime(filename) == initial_mtime
    assert path == filename

def test_generate_all_stimuli(stimulus_generator):
    """Test that generate_all_stimuli creates all stimulus combinations."""
    # Clear directory
    for file in os.listdir(ExperimentConfig.IMAGE_DIR):
        os.remove(os.path.join(ExperimentConfig.IMAGE_DIR, file))
    
    # Generate all stimuli
    stimulus_generator._generate_all_stimuli()
    
    # Verify all combinations were created
    expected_files = set()
    for word, color in ExperimentConfig.STIMULI:
        expected_files.add(f"{word}_{color[0]}_{color[1]}_{color[2]}.png")
    
    actual_files = set(os.listdir(ExperimentConfig.IMAGE_DIR))
    assert actual_files == expected_files 