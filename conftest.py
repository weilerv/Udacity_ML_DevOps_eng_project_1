import pytest

def df_plugin():
    return None

def  X_train_plugin():
    return None

def  X_test_plugin():
    return None

def  y_train_plugin():
    return None

def  y_test_plugin():
    return None

# Creating a Dataframe object 'pytest.df' in Namespace
def pytest_configure():
    pytest.df = df_plugin()
    pytest.X_train = X_train_plugin()
    pytest.X_test = X_test_plugin()
    pytest.y_train = y_train_plugin()
    pytest.y_test = y_test_plugin()