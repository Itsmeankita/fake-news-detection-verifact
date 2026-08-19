"""
Promotes an existing user to admin (needed to view the /admin panel).

Usage:
  1. Sign up normally on the website first (creates your account).
  2. Run: python make_admin.py your_username
"""
import sys

from app import app, db, User

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_admin.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        if not user:
            print(f"No user found with username '{username}'. Sign up on the site first.")
            sys.exit(1)
        user.is_admin = True
        db.session.commit()
        print(f"'{username}' is now an admin. Log in and visit /admin.")
