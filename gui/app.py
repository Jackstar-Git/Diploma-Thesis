from flask import Flask, render_template, request, redirect, url_for, session, abort
import os
import json
import sys
from modules.helpers import fetch_books_api
import threading
import time
import requests

app_root = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(app_root, "templates")
static_folder = os.path.join(app_root, "static")


app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":    
        preferences = (request.form.get("pages", None) ,request.form.get("author", ""), request.form.get("genres", ""), request.form.get("story_elements", ""), request.form.get("description", ""))
        #pages, author_input, genres_input, story_elements_input, description_input
        results = fetch_books_api(preferences, int(request.form.get("results", 5)))

        return render_template("index.jinja-html", results=results, preferences=preferences)
    return render_template("index.jinja-html", results=None)

@app.route("/settings", methods=["GET", "POST"])
def settings():
    with open("settings.json", "r") as f:
        weights = json.load(f)
        print(weights, file=sys.stderr)

    if request.method == "POST":
        # if sum not exactly 1.0, return error, save every value a variable first
        description_weight = float(request.form.get("description_weight", 15))
        author_weight = float(request.form.get("author_weight", 20))
        genres_weight = float(request.form.get("genres_weight", 45))
        story_elements_weight = float(request.form.get("story_elements_weight", 15))
        page_length_weight = float(request.form.get("page_length_weight", 5))
        total_weight = description_weight + author_weight + genres_weight + story_elements_weight + page_length_weight
        if total_weight != 100:
            message = "The sum of all weights must be exactly 100%! Current sum: {:.2f}".format(total_weight)
            return render_template("settings.jinja-html", message=message, weights=weights)

        with open("settings.json", "w") as f:
            json.dump({
                "description": description_weight/100,
                "author": author_weight/100,
                "genre": genres_weight/100,
                "story_element": story_elements_weight/100,
                "page_length": page_length_weight/100
            }, f, indent=4)

        message = "Settings updated!"
        return render_template("settings.jinja-html", message=message, weights=weights)
    return render_template("settings.jinja-html", message=None, weights=weights)

@app.route("/about")
def about():
    return render_template("about.jinja-html")

@app.route("/info")
def info():
    return abort(404)
    #return render_template("info.jinja-html")




