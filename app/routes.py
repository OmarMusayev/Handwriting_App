# app/routes.py
from flask import (
    request, render_template, flash, redirect,
    jsonify, url_for, session
)
from app import flask_app  # from your __init__.py
import os
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

load_dotenv()

# Configure the Flask app
flask_app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
flask_app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "fallback_secret")

# Configure disk storage path
DISK_STORAGE_PATH = "/disk"  # Render's disk storage path
UPLOADS_PATH = os.path.join(DISK_STORAGE_PATH, "uploads")
# Ensure the uploads directory exists
os.makedirs(UPLOADS_PATH, exist_ok=True)

###################################
# 1) Define db & bcrypt FIRST
###################################
db = SQLAlchemy(flask_app)
bcrypt = Bcrypt(flask_app)

###################################
# 2) THEN import models AFTER db is defined
###################################
from app.models import User, SavedSample


# Existing imports for your handwriting logic, etc.
import base64
import uuid
import numpy as np
from app.priming import generate_handwriting
from app.xml_parser import svg_xml_parser, path_to_stroke, path_string_to_stroke
from utils import plot_stroke


# ------------------------------------------------------------------
# SIGN UP
# ------------------------------------------------------------------
@flask_app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        # Check if username or email is already taken
        existing_username = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()

        if existing_username:
            flash("Username already in use. Please choose another.", "warning")
            return redirect(url_for("signup"))

        if existing_email:
            flash("Email already in use. Please choose another.", "warning")
            return redirect(url_for("signup"))

        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)  
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful! You can now log in.", "success")
        return redirect(url_for("login"))
    else:
        return render_template("signup.html", title="Sign Up")



# ------------------------------------------------------------------
# LOGIN
# ------------------------------------------------------------------
@flask_app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            session["username"] = user.username  # store the username in session
            flash("Logged in successfully!", "success")
            return redirect(url_for("profile"))
        else:
            flash("Invalid credentials. Please try again.", "danger")
            return redirect(url_for("login"))
    else:
        return render_template("login.html", title="Login")

# ------------------------------------------------------------------
# LOGOUT
# ------------------------------------------------------------------
@flask_app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ------------------------------------------------------------------
# ABOUT / HOME 
# ------------------------------------------------------------------
@flask_app.route("/", methods=["GET"])
@flask_app.route("/login", methods=["GET"])
def index():
    if "user_id" in session:
        # If the user is logged in, go straight to profile
        return redirect(url_for("profile"))
    else:
        # Otherwise, show the login page
        return render_template("login.html", title="Login")


# ------------------------------------------------------------------
# DRAW (WRITE) 
# ------------------------------------------------------------------
@flask_app.route("/draw", methods=["GET"])
def draw():
    if "id" in session:
        print("uuid: ", session["id"])
    return render_template("draw.html", title="Write")


# ------------------------------------------------------------------
# UPLOAD STYLE 
# ------------------------------------------------------------------
@flask_app.route("/upload_style", methods=["POST"])
def submit_style_data():
    try:
        data = request.get_json()
        path = data.get("path", "")
        text = data.get("text", "")
        
        if not path:
            return jsonify({"redirect": url_for("draw"), "message": "Please enter some style"})

        id = str(uuid.uuid4())
        session["id"] = id
        
        # Use the disk storage path
        tmp_dir = os.path.join(UPLOADS_PATH, id)
        os.makedirs(tmp_dir, exist_ok=True)
        os.chmod(tmp_dir, 0o777)

        # Save input text
        text_path = os.path.join(tmp_dir, "inpText.txt")
        with open(text_path, "w") as f:
            f.write(text)

        # Process and save stroke data
        stroke = path_string_to_stroke(
            path, str_len=len(list(text)), down_sample=True
        )
        save_path = os.path.join(tmp_dir, "style.npy")
        np.save(save_path, stroke, allow_pickle=True)

        # Generate preview image
        plot_stroke(stroke.astype(np.float32), os.path.join(tmp_dir, "original.png"))

        return jsonify({"redirect": url_for("generate"), "message": ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# GENERATE (HANDWRITING) 
# ------------------------------------------------------------------
@flask_app.route("/generate", methods=["GET", "POST"])
def generate():
    # Default style paths now point to disk storage
    default_style_path = os.path.join(flask_app.root_path, "static/uploads/default_style.npy")
    # Encode default image
    with open(os.path.join(flask_app.root_path, "static/uploads/default.png"), "rb") as imgfile:
        org_img = base64.b64encode(imgfile.read())
    org_src = "data:image/png;base64,{}".format(org_img.decode("ascii"))

    if request.method == "POST":
        text = request.form.get("text", "")
        bias = float(request.form.get("bias", 1.0))
        style_option = request.form.get("styleOptions", "defaultStyle")
        
        if not text:
            return render_template(
                "generate.html",
                title="Generate",
                message="Please enter some text",
                text="",
                org_src=org_src,
                samples="",
            )

        # If user picks default style
        if style_option == "defaultStyle":
            style_path = default_style_path
            real_text = "copy monkey app"
            # Ensure session ID
            if "id" not in session:
                session["id"] = str(uuid.uuid4())

            tmp_dir = os.path.join(UPLOADS_PATH, session["id"])
            os.makedirs(tmp_dir, exist_ok=True)
            os.chmod(tmp_dir, 0o777)

        # If user picks your style but no session ID
        elif "id" not in session:
            return render_template(
                "generate.html",
                title="Generate",
                text="",
                message="Please go to Write and add some style.",
                org_src=org_src,
            )

        # If user picks your style and session is set
        elif style_option == "yourStyle":
            tmp_dir = os.path.join(UPLOADS_PATH, session["id"])
            style_path = os.path.join(tmp_dir, "style.npy")
            if not os.path.exists(style_path):
                return render_template(
                    "generate.html",
                    title="Generate",
                    text="",
                    message="Please go to Write and add some style.",
                    org_src=org_src,
                )
                
            org_img_path = os.path.join(tmp_dir, "original.png")
            if os.path.exists(org_img_path):
                with open(org_img_path, "rb") as imgfile:
                    org_img = base64.b64encode(imgfile.read())
                org_src = "data:image/png;base64,{}".format(org_img.decode("ascii"))

            text_path = os.path.join(tmp_dir, "inpText.txt")
            with open(text_path) as file:
                texts = file.read().splitlines()
            real_text = texts[0]

        # Now generate handwriting
        save_path = os.path.join(tmp_dir)
        n_samples = 5

        generate_handwriting(
            char_seq=text,
            real_text=real_text,
            style_path=style_path,
            save_path=save_path,
            app_path=flask_app.root_path,
            n_samples=n_samples,
            bias=bias,
        )

        # Gather the generated images
        gen_samp = []
        for i in range(n_samples):
            path = os.path.join(save_path, f"gen_stroke_{i}.png")
            with open(path, "rb") as genfile:
                encoded_image = base64.b64encode(genfile.read())
            src = f"data:image/png;base64,{encoded_image.decode('ascii')}"
            gen_samp.append(src)

        return render_template(
            "generate.html",
            title="Generate",
            text=text,
            bias=bias,
            org_src=org_src,
            samples=gen_samp,
        )

    # GET request
    else:
        if "id" in session:
            tmp_dir = os.path.join(UPLOADS_PATH, session["id"])
            org_img_path = os.path.join(tmp_dir, "original.png")
            if os.path.exists(org_img_path):
                with open(org_img_path, "rb") as imgfile:
                    org_img = base64.b64encode(imgfile.read())
                org_src = "data:image/png;base64,{}".format(org_img.decode("ascii"))

        return render_template(
            "generate.html",
            title="Generate",
            text="",
            org_src=org_src,
            samples="",
            message="",
        )

@flask_app.route("/profile")
def profile():
    if "user_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))
    
    user_id = session["user_id"]
    # Query all saved samples for this user
    my_samples = SavedSample.query.filter_by(user_id=user_id).all()

    return render_template("profile.html", samples=my_samples, title="Profile")


@flask_app.route("/save_sample", methods=["POST"])
def save_sample():
    if "user_id" not in session:
        flash("Please log in to save samples.", "warning")
        return redirect(url_for("login"))

    image_data = request.form.get("image_data")
    if not image_data:
        flash("No image data provided. Unable to save.", "danger")
        return redirect(url_for("generate"))

    # Save to disk storage if it's a file path, not just an image URI
    if "id" in session and not image_data.startswith("data:image"):
        user_id = session["user_id"]
        user_dir = os.path.join(UPLOADS_PATH, f"user_{user_id}")
        os.makedirs(user_dir, exist_ok=True)
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        image_path = os.path.join(user_dir, f"{file_id}.png")
        
        # Copy the file from temp directory to user directory
        with open(image_data, "rb") as src_file:
            with open(image_path, "wb") as dst_file:
                dst_file.write(src_file.read())
                
        # Store the path in database
        image_data = image_path

    new_sample = SavedSample(
        user_id=session["user_id"],
        image_path=image_data  
    )
    db.session.add(new_sample)
    db.session.commit()

    flash("Sample saved to your profile!", "success")
    return redirect(url_for("profile"))
