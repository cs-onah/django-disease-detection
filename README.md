# Django Disease Detection

Django Disease Detection is a web application that uses machine learning to detect diseases in images.

## Installation

1. Clone the repository:

```bash
$ git clone https://github.com/your-username/django-disease-detection.git
$ cd django-disease-detection
```

2. Install dependencies:

```bash
$ pip install -r requirements.txt
```

3. Run the development server:

```bash
$ gunicorn cassava_disease_detection.wsgi:application
```

4. Access the application at http://localhost:8000.

## Usage

1. Upload an image of a disease.
2. The application will analyze the image and return the results.

## License

Django Disease Detection is licensed under the MIT License.
