function readMore(element) {
    var dots = element.querySelector("#dots");
    var moreButton = element.querySelector(".more-button");
    var moreText = element.querySelector("#read-more");
    if (moreText.style.display === "none") {
        moreText.style.display = "inline";
        dots.style.display = "none";
        moreButton.classList.add("enabled");
    } else {
        moreText.style.display = "none";
        moreButton.classList.remove("enabled");
        dots.style.display = "inline";
    }
};

document.addEventListener("DOMContentLoaded", function() {
    var readMoreElements = document.querySelectorAll(".results-table tr .more-button");
    readMoreElements.forEach(function(element) {
        element.addEventListener("click", function() {
            readMore(element.parentElement);
        });
    });
});