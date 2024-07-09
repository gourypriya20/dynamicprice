$(document).ready(function() {
    $("#predict-button").click(function() {
      // Get product description from the text area
      var description = $("#description").val();
  
      // Send AJAX POST request to predict price
      $.ajax({
        url: "/predict_price",
        type: "POST",
        data: { description: description },
        dataType: "json",
        success: function(response) {
          var predictedPrice = response.predicted_price;
          var fraudClassification = response.fraud_classification;
  
          // Update the result element with prediction and classification
          $("#result").html("Predicted Price: $" + predictedPrice + "<br>Fraud Classification: " + fraudClassification);
        },
        error: function(jqXHR, textStatus, errorThrown) {
          console.error("Error:", textStatus, errorThrown);
          $("#result").html("An error occurred during prediction.");
        }
      });
    });
  });
  