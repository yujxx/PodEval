function LikertScale2(_responseConfig, _prefix, _initDisabled, _callback) {
  this.responseConfig = _responseConfig;
  this.prefix = _prefix;
  this.initDisabled = _initDisabled;
  this.callback = _callback;  
  this.group = null;
  this.elements = [];
} 

LikertScale2.prototype.enable = function () {
  $('input[name='+this.prefix+'_response]').checkboxradio('enable');
}

LikertScale2.prototype.render = function (_parent) {
  this.group = $("<fieldset data-role='controlgroup' data-type='horizontal'></fieldset>");
  _parent.append(this.group);
  this.elements = [];
  
  if(this.responseConfig != null){
    for (var i = 0; i < this.responseConfig.length; ++i) {
      var responseValueConfig = this.responseConfig[i];
      var img = "";
      if (responseValueConfig.title){
        var header = $("<p class='session-header'>" + responseValueConfig.title + "</p>");
        this.group.append(header);
        if (responseValueConfig.description){
          var description = $("<p class='session-description'>" + responseValueConfig.description + "</p></br>");
          this.group.append(description);
        }
      }

      if (responseValueConfig.type == "checkbox"){
        var element = $("<p>" + responseValueConfig.label + "</p>");
        this.group.append(element);
        if (responseValueConfig.options){
          var options = responseValueConfig.options.split('/');
          var container = $("<div class='checkbox-container'></div>");
          
          for (var j = 0; j < options.length; ++j) {
            var checkbox = $("<input type='checkbox' class='custom-checkbox' data-mini='false' value='"+j+"' name='"+this.prefix+"_response' id='"+this.prefix+"_response2_"+j+"'/>");
            var label = $("<label for='"+this.prefix+"_response2_"+j+"' class='custom-label'>"+options[j]+"</label>");
            container.append(checkbox, label);
            this.elements.push(checkbox, label);
          }

          this.group.append(container);
        }
        this.group.append($("</br>"));
      } else if (responseValueConfig.type == "option"){
        var element = $("<p>" + responseValueConfig.label + "</p>");
        this.group.append(element);
        var element = $("<input id='comment_option'></input>");
        this.group.append(element);

        
      } else {
        var img = "";
        if(responseValueConfig.img) {
          img = "<img id='"+this.prefix+"_response_img_"+i+"' src='"+responseValueConfig.img+"'/>";
        }
        var element = $("<input type='checkbox' data-mini='false' value='"+responseValueConfig.value+"' name='"+this.prefix+"_response' id='"+this.prefix+"_response2_"+i+"'/> \
          <label for='"+this.prefix+"_response2_"+i+"' class='custom-label2'>"+responseValueConfig.label +"</br>"+img +"</label> \
        ");
        this.group.append(element);
        this.elements[this.elements.length] = element;
      }
    }

    this.group.change((function() {
      
      if (this.callback) {
        this.callback(this.prefix);
      }
    }).bind(this));
  }
  if (this.initDisabled) {
    this.group.children().attr('disabled', 'disabled');    
  }
  
};