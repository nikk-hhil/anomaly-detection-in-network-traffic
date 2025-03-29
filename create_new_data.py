import os
import sys
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Create a test dataset from the original merged data."""
    logger.info("Starting test data creation...")
    
    # Set up paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Project directory: {project_dir}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Possible locations for the merged data
    potential_paths = [
        os.path.join(project_dir, "data", "processed", "merged_data.csv"),
        os.path.join(project_dir, "data", "merged_data.csv"),
        os.path.join(os.path.dirname(project_dir), "data", "processed", "merged_data.csv"),
        os.path.join(os.path.dirname(project_dir), "data", "merged_data.csv")
    ]
    
    # Try to find the merged data
    merged_data_path = None
    for path in potential_paths:
        logger.info(f"Checking for merged data at: {path}")
        if os.path.exists(path):
            merged_data_path = path
            logger.info(f"Found merged data at: {path}")
            break
    
    # If merged data not found, ask for the path
    if merged_data_path is None:
        logger.warning("Merged data not found in expected locations.")
        print("\nPlease enter the full path to your merged_data.csv file:")
        merged_data_path = input().strip()
        
        if not os.path.exists(merged_data_path):
            logger.error(f"File not found: {merged_data_path}")
            sys.exit(1)
    
    # Create test data
    test_data_path = os.path.join(data_dir, "test_data.csv")
    
    try:
        logger.info(f"Loading data from: {merged_data_path}")
        # Read the first 1000 rows
        test_data = pd.read_csv(merged_data_path, nrows=1000)
        logger.info(f"Successfully loaded {len(test_data)} rows")
        
        # Save test data
        logger.info(f"Saving test data to: {test_data_path}")
        test_data.to_csv(test_data_path, index=False)
        logger.info(f"Successfully created test data with {len(test_data)} rows")
        
        # Print some statistics
        logger.info("\nTest data statistics:")
        logger.info(f"Shape: {test_data.shape}")
        
        if ' Label' in test_data.columns:
            logger.info("\nClass distribution:")
            class_dist = test_data[' Label'].value_counts()
            for cls, count in class_dist.items():
                logger.info(f"  {cls}: {count} ({count/len(test_data)*100:.2f}%)")
        
        logger.info("\nTest data creation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error creating test data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()