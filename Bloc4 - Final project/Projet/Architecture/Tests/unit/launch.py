#https://openclassrooms.com/fr/courses/7155841-testez-votre-projet-python/7414156-ajoutez-des-tests-avec-pytest
from test_ import reverse_str

separator = '=' * 80
print(separator)
print("Running unit tests for launch.py")
print(separator)

def test_should_reverse_string():
    assert reverse_str('abc') == 'cba'