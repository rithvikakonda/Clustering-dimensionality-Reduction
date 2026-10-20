# Cell 2: All Imports Required
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Cell 3: StudentDataset Class Definition

class StudentDataset:
    def __init__(self, num_students: int, seed: int):

        """
        Create a dataset of students with randomly generated attributes.
        Initializes the StudentDataset object and generates the dataset.
        Parameters:
            num_students (int):  Number of student records to generate.
            seed (int): Random seed so results are reproducible.
        """
        self.num_students = num_students
        self.seed = seed
          # Random number generator (for consistent random results)
        self.rng = np.random.default_rng(self.seed)
          # Generate student attributes
        self.gender = self.generate_gender()
        self.major = self.generate_major()
        self.program = self.generate_program(self.major)
        self.gpa = self.generate_gpa(self.major)
          # Combine into a pandas DataFrame
        self.df = self.assemble_dataframe()

    def generate_gender(self) -> list[str]:
        """
        Randomly assign a gender to each student, based on probability.
        Returns:
            List of genders for all students.
        """
        
        gender_options  = ['Male', 'Female', 'Other']
        probabilities = [0.65, 0.33, 0.02]  # 65% male, 33% female, 2% other
        
        genders = self.rng.choice(
            gender_options,
            size=self.num_students,
            p=probabilities
        )

        return genders.tolist()  # convert from NumPy array → Python list


    def generate_major(self) -> list[str]:
        """
        Generates major for each student based on specified probabilities.
        Returns:
            list[str]: List of majors.
        """
        major_options = ['B.Tech', 'MS', 'PhD']
        probs = [0.70, 0.20, 0.10]
        majors = self.rng.choice(
            major_options,
            size=self.num_students,
            p=probs
        )
        
        
        return majors.tolist()  # con
            # 'Program': self.progravert from NumPy array → Python list

    def generate_program(self, majors: list[str]) -> list[str]:
        """
        Assign a program specialization depending on the student's major.
        Args:
            majors (list[str]): List of majors for each student.
        Returns:
            list[str]: List of programs.
        """
      
        program_list = []
        for major in majors:
            if major == 'B.Tech':
                choices = ['CSE', 'ECE', 'CHD', 'CND']
                probs = [0.4, 0.4, 0.1, 0.1]
            elif major == 'MS':
                choices = ['CSE', 'ECE', 'CHD', 'CND']
                probs = [0.3, 0.3, 0.2, 0.2]
            else:  # PhD
                choices = ['CSE', 'ECE', 'CHD', 'CND']
                probs = [0.25, 0.25, 0.25, 0.25]
                
            program = self.rng.choice(choices, p=probs)
            program_list.append(program)
           
        return program_list

    def generate_gpa(self, majors: list[str]) -> list[float]:
        """
        Generates GPA for each student based on their major, clipped to [4.0, 10.0].
        Args:
            majors (list[str]): List of majors for each student.
        Returns:
            list[float]: List of GPAs.
        """
        gpa_list = []
        for major in majors:
            if major == 'B.Tech':
                gpa = self.rng.normal(7.0, 1.0)
            elif major == 'MS':
                gpa = self.rng.normal(8.0, 0.7)
            else:  # PhD
                gpa = self.rng.normal(8.3, 0.5)
             # Clip GPA between 4.0 and 10.0
            gpa = float(np.clip(gpa, 4.0, 10.0))
            gpa_list.append(gpa)

            
        return gpa_list

    def assemble_dataframe(self) -> pd.DataFrame:
        """
        Combine all student attributes into a DataFrame
        Returns:
            pd.DataFrame: DataFrame containing all student records.
        """
        
        data = {
            'Gender': self.gender,
            'Major': self.major,
            'Program': self.program,
            'GPA': self.gpa
        }
        return pd.DataFrame(data)

    def get_full_dataframe(self) -> pd.DataFrame:
        """
        Returns the full student dataset as a pandas DataFrame.
        Returns:
            pd.DataFrame: Full dataset.
        """
        return self.df
    
    
    #  Q1.1 
    
    