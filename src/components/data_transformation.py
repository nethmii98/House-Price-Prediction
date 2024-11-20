import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import Normalizer, MinMaxScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation

        '''


        try:
            numerical_columns = ['sqft_living','sqft_lot','sqft_basement','yr_built','lat','long']
            categorical_columns = ['bedrooms','bathrooms','floors','condition','grade']

            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                  ("imputer",SimpleImputer(strategy="most_frequent")),
                  ("one_hot_encoder",OneHotEncoder()),
                  ("scaler",StandardScaler(with_mean=False))  
                ]
            )    
            
            logging.info("numerical columns standard scaling completed ")
            logging.info("Categorical columns encoding completed ")

            preprocessor=ColumnTransformer(
               [
                  ("num_pipeline",num_pipeline,numerical_columns),
                  ("cat_pipeline",cat_pipeline,categorical_columns)  
               ] 
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys) 
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            #train_df.drop(columns=['id','date','waterfront','view','sqft_above','yr_renovated','zipcode','sqft_living15','sqft_lot15'],inplace=True)
            #test_df.drop(columns=['id','date','waterfront','view','sqft_above','yr_renovated','zipcode','sqft_living15','sqft_lot15'],inplace=True)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'price'
            numerical_columns = ['sqft_living','sqft_lot','sqft_basement','yr_built','lat','long']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            # input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df).toarray()
            scaler = MinMaxScaler()

            input_feature_train_arr = scaler.fit_transform(input_feature_train_df)
            input_feature_test_arr = scaler.fit_transform(input_feature_test_df)

            logging.info(f"Shapes after transformation: X_train={input_feature_train_arr.shape}, X_test={input_feature_test_arr.shape}")


            print(type(input_feature_train_arr))
            print(type(target_feature_train_df))    

            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_df: {np.array(target_feature_train_df).shape}")
            from scipy.sparse import csr_matrix,hstack

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
            
            logging.info(f"Saved preprocessing object.")

            logging.info(f"Shape of train_arr: {train_arr.shape}")
            logging.info(f"Shape of test_arr: {test_arr.shape}")

            # save_object(
            #     file_path=self.data_transformation_config.preprocessor_obj_file_path,
            #     obj = preprocessing_obj
            # )

            return(
                train_arr,
                test_arr,
            )

        except Exception as e:
            raise CustomException(e,sys)
            