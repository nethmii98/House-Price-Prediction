import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix,hstack


from src.exception import CustomException
from src.logger import logging

@dataclass

class DataTransformation:
    
        
    def initiate_data_transformation(self,train_path,test_path):
        try:

            train_df = pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            target_column_name = 'price'

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            scaler = StandardScaler()

            input_feature_train_arr = scaler.fit_transform(input_feature_train_df)
            input_feature_test_arr = scaler.fit_transform(input_feature_test_df)


            logging.info(f"Shapes after transformation: X_train={input_feature_train_arr.shape}, X_test={input_feature_test_arr.shape}")


            print(type(input_feature_train_arr))
            print(type(target_feature_train_df))    

            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_df: {np.array(target_feature_train_df).shape}")

            target_train_array = target_feature_train_df.values

            # Create a sparse matrix from the Series
            target_train_sparse = csr_matrix(target_train_array).reshape(-1, 1)  # Reshape to 2D

            # Concatenate the sparse matrix and the target variable
            train_arr = hstack([input_feature_train_arr, target_train_sparse])
            
            
            target_test_array = target_feature_test_df.values

            # Create a sparse matrix from the Series
            target_test_sparse = csr_matrix(target_test_array).reshape(-1, 1)  # Reshape to 2D

            # Concatenate the sparse matrix and the target variable
            test_arr = hstack([input_feature_test_arr, target_test_sparse])
            

            logging.info(f"Shape of train_arr: {train_arr.shape}")
            logging.info(f"Shape of test_arr: {test_arr.shape}")


            return(
                train_arr,
                test_arr,
            )

        except Exception as e:
            raise CustomException(e,sys)
            