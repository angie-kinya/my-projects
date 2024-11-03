from django.shortcuts import render
from .models import Project, Skill

def portfolio_home(request):
    projects = Project.objects.all()
    skills = Skill.objects.all()
    return render(request, 'portfolio/home.html', {'projects':projects, 'skills':skills})
