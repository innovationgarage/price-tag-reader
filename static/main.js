function uploadImage(data) {
  var form = new FormData();
  form.append("file", data);
  var req = $.post("/analyze", { data: data });
  req.done(function(data){
    alert(data);
  });
}

function capture() {
  var constraints = {
    video: {
      width: { min: 1280, ideal: 1280, max: 1920 },
      height: { min: 720, ideal: 720, max: 1080 },
      facingMode: "environment"
    }
  };

  var later = navigator.mediaDevices.getUserMedia(constraints);
  later.then(function(stream) {
    var video = document.getElementById("video");
    var fullsize = document.createElement('video');
    var button = $("#b");
    var canvas;
    var videoWidth = 0
    var videoHeight = 0;

    video.srcObject = stream;
    video.play();
    video.onloadedmetadata = function() {
      videoHeight = this.videoHeight;
      videoWidth = this.videoWidth;

      fullsize.width = videoWidth;
      fullsize.height = videoHeight;
      fullsize.srcObject = stream.clone();
      fullsize.play();
    }

    button.removeAttr("disabled");
    button.click(function() {
      canvas = canvas || document.createElement('canvas');
      canvas.width = videoWidth;
      canvas.height = videoHeight;
      canvas.getContext("2d").drawImage(fullsize, 0, 0, videoWidth, videoHeight);

      var img = canvas.toDataURL("image/png");
      uploadImage(img);
    });
  }).catch(function(err) {
    alert("there was an error " + err);
  });
}

$(document).ready(capture);