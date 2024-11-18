from django.urls import path
from . import views

urlpatterns = [
    path('get-plan/', views.get_personalized_plan, name='get_plan'),
]