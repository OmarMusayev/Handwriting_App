# create_tables.py

from app import flask_app, db

# import your models so they’re registered with SQLAlchemy
from app.models import User, SavedSample

with flask_app.app_context():
    db.create_all()
    print("✅ Created all tables:")
    for table in db.metadata.tables:
        print("   -", table)
