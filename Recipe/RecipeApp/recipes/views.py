from django.shortcuts import render, get_object_or_404, redirect
from .models import Recipe

# Create your views here.
def recipe_list(request):
    recipes = Recipe.objects.all()
    return render(request, 'recipes/recipe_list.html', {'recipes':recipes})

def recipe_detail(request, id):
    recipe = get_object_or_404(Recipe, id=id)
    return render(request, 'recipes/recipe_detail.html', {'recipe':recipe})

def add_recipe(request):
    if request.method =='POST':
        title = request.POST.get('title')
        ingredients = request.POST.get('ingredients')
        instructions = request.POST.get('instructions')
        Recipe.objects.create(title=title, ingredients=ingredients, instructions=instructions)
        return redirect('recipe_list')
    return render(request, 'recipes/add_recipe.html')