#!/bin/bash

# Exit if any command fails
set -e

echo "📦 Applying migrations..."
python manage.py makemigrations
python manage.py migrate

# Create superuser if it doesn't exist
echo "👤 Checking for existing superuser..."
python manage.py shell <<EOF
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username="admin").exists():
    print("🔑 Creating superuser: admin / admin123")
    User.objects.create_superuser("admin", "admin@example.com", "admin123")
else:
    print("✅ Superuser already exists")
EOF

# Start development server
echo "🚀 Starting development server..."
python manage.py runserver