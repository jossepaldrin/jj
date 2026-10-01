I cannot directly connect to your GitHub account or create a repository on your behalf since I don't have access to your personal authentication credentials (like a GitHub Personal Access Token or SSH keys).

However, I can provide you with the **complete, exam-ready codebase and guide** that you can copy-paste directly into a new GitHub repository named `fsd-exam-cheatsheet` (or any name you choose under your `jossepaldrin` account).

Here is everything you need for your Full Stack Development (FSD) exam—covering project setup, views, models, ModelForms, custom forms, authentication, login, and logout—packaged into a comprehensive Markdown file (`README.md`).

---

### Step 1: Create the Repository on GitHub

1. Go to [GitHub](https://github.com/) and log in as **jossepaldrin**.
2. Click the **`+`** icon in the top-right corner and select **New repository**.
3. Name your repository `fsd-exam-guide` (or similar).
4. Set it to **Public** or **Private**, check **Add a README file**, and click **Create repository**.
5. Edit the `README.md` file directly on GitHub and paste the content below into it.

---

### Step 2: Copy-Paste This README Content into Your GitHub Repo

```markdown
# 🚀 Ultimate Django FSD Exam Cheatsheet & Guide

This repository contains everything you need to ace your Full Stack Development (FSD) exam using Django. From project setup to advanced authentication and forms, use this as your ultimate quick-reference guide.

---

## 📌 Table of Contents
1. [Project Setup & Configuration](#1-project-setup--configuration)
2. [Database Models (`models.py`)](#2-database-models-modelspy)
3. [Forms & ModelForms (`forms.py`)](#3-forms--modelforms-formspy)
4. [Views (`views.py`)](#4-views-viewspy)
5. [URL Routing (`urls.py`)](#5-url-routing-urlspy)
6. [HTML Templates (`templates/`)](#6-html-templates)
7. [Authentication, Login & Logout](#7-authentication-login--logout)

---

## 1. Project Setup & Configuration

### Commands to Initialize a Project
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Django
pip install django

# Start project and app
django-admin startproject myproject .
python manage.py startapp myapp

```

### Essential `settings.py` Additions

Always register your app in `INSTALLED_APPS` and configure static/template paths if necessary:

```python
INSTALLED_APPS = [
    # ... default apps ...
    'myapp',
]

# Redirect URLs after login/logout
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'home'
LOGOUT_REDIRECT_URL = 'login'

```

---

## 2. Database Models (`models.py`)

Models define your database schema. Always inherit from `models.Model`.

```python
from django.db import models
from django.contrib.auth.models import User

class Post(models.Model):
    # Foreign key to Django's built-in User model (Cascade deletes posts if user is deleted)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title

```

**Common Field Types to Remember:**

* `models.CharField(max_length=100)` — Short text
* `models.TextField()` — Long paragraphs/text
* `models.IntegerField()` — Numbers
* `models.EmailField()` — Validated email address
* `models.DateField()`, `models.DateTimeField()` — Dates/Times

Run migrations after creating or modifying models:

```bash
python manage.py makemigrations
python manage.py migrate

```

---

## 3. Forms & ModelForms (`forms.py`)

### A. ModelForm (Automatically maps to a Model)

Use this when you want to save data directly into the database via a model.

```python
from django import forms
from .models import Post

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'content']
        # Optional: Add CSS widgets
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter title'}),
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 5}),
        }

```

### B. Standard Form (For Custom Input / Login / Contact)

Use this when data doesn't map directly to a single model (e.g., Search, Contact forms).

```python
from django import forms

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()
    message = forms.CharField(widget=forms.Textarea)

```

---

## 4. Views (`views.py`)

Views handle the business logic, process requests, and return responses.

```python
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.decorators import login_required
from .models import Post
from .forms import PostForm

# 1. Home / List View
def home_view(request):
    posts = Post.objects.all().order_by('-created_at')
    return render(request, 'myapp/home.html', {'posts': posts})

# 2. Create Post View (Protected by login_required)
@login_required
def create_post(request):
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user  # Attach currently logged-in user
            post.save()
            return redirect('home')
    else:
        form = PostForm()
    return render(request, 'myapp/post_form.html', {'form': form})

# 3. User Signup / Registration View
def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log user in immediately after registration
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'myapp/register.html', {'form': form})

# 4. Login View
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'myapp/login.html', {'form': form})

# 5. Logout View
def logout_view(request):
    logout(request)
    return redirect('login')

```

---

## 5. URL Routing (`urls.py`)

### App-level `myapp/urls.py`

```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('create/', views.create_post, name='create_post'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]

```

### Project-level `myproject/urls.py`

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('myapp.urls')),
]

```

---

## 6. HTML Templates (`templates/myapp/`)

### Essential Template Tags to Remember:

* `{% csrf_token %}` — Mandatory inside **every** `<form>` method="POST".
* `{{ form.as_p }}` — Renders form fields inside paragraphs.
* `{% url 'home' %}` — Dynamic URL resolution.

### Example Form Template (`post_form.html` or `login.html`)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Form</title>
</head>
<body>
    <h2>Submit Form</h2>
    <form method="POST">
        {% csrf_token %}
        {{ form.as_p }}
        <button type="submit">Submit</button>
    </form>
    <a href="{% url 'home' %}">Back to Home</a>
</body>
</html>

```

---

## 7. Authentication Quick Reference

| Feature | Built-in Tool / Function | Import Path |
| --- | --- | --- |
| **Password Hashing & User Creation** | `UserCreationForm` | `from django.contrib.auth.forms import UserCreationForm` |
| **Login Authentication Form** | `AuthenticationForm` | `from django.contrib.auth.forms import AuthenticationForm` |
| **Log User In** | `login(request, user)` | `from django.contrib.auth import login` |
| **Log User Out** | `logout(request)` | `from django.contrib.auth import logout` |
| **Protect Views (Decorator)** | `@login_required` | `from django.contrib.auth.decorators import login_required` |
| **Access Current User** | `request.user` | Available automatically in views & templates |
| **Check if Logged In** | `request.user.is_authenticated` | Available in templates & views |

```

---

### Quick Exam Checklist:
1. Did you include `{% csrf_token %}` in your POST forms? **(Most common exam mistake if missed)**
2. Did you set `request.user` before saving a model linked to an author? (`post.author = request.user`)
3. Are all your app URLs included in the main project `urls.py`?

```
