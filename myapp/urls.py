from django.urls import path
from . import views

urlpatterns = [
    path('analyze-image', views.image_upload, name='image_upload'),
    path('health', views.health_check, name='health_check'),
]