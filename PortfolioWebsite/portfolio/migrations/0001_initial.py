# Generated by Django 5.1.2 on 2024-11-03 10:03

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('description', models.TextField()),
                ('technologies', models.CharField(max_length=200)),
                ('github_link', models.URLField(blank=True, null=True)),
                ('live_link', models.URLField(blank=True, null=True)),
                ('image', models.ImageField(blank=True, null=True, upload_to='project_images/')),
                ('creted_at', models.DateField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Skill',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('level', models.CharField(max_length=20)),
            ],
        ),
    ]
