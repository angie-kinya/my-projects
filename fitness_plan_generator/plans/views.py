from django.shortcuts import render
from django.http import JsonResponse
from .models import FitnessPlan

def get_personalized_plan(request):
    # logic for generating a personalized plan
    plan = {"message": "Sample fitness plan data"}
    return JsonResponse(plan)