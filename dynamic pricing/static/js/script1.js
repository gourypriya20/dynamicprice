function resetClass(element, classname){
    element.classList.remove(classname);
  }
  document.getElementsByClassName("show-signup")[0].addEventListener("click",function(){
    let form = document.getElementsByClassName("form")[0];
    resetClass(form, "signin");
    resetClass(form, "reset");
    form.classList.add("signup");
    document.getElementById("submit-btn").innerText = "Sign Up";
  });
  document.getElementsByClassName("show-signin")[0].addEventListener("click",function(){
    let form = document.getElementsByClassName("form")[0];
    resetClass(form, "signup");
    resetClass(form, "reset");
    form.classList.add("signin");
    document.getElementById("submit-btn").innerText = "Sign In";
  });

  document.getElementsByClassName("show-reset")[0].addEventListener("click",function(){
    let form = document.getElementsByClassName("form")[0];
    resetClass(form, "signup");
    resetClass(form, "signin");
    form.classList.add("reset");
    document.getElementById("submit-btn").innerText = "Reset password";
  });


  document.getElementById('signInForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form from submitting the default way
    let username = document.getElementById('username').value;
    let password = document.getElementById('password').value;
    let messageDiv = document.getElementById('message');
    if (username === '' || password === '') {
        messageDiv.textContent = 'Please fill out both fields.';
        messageDiv.style.color = 'red';
    } else {
        this.submit(); // If validation passes, submit the form
    }
});

