/*************************************************************************
         (C) Copyright AudioLabs 2017 

This source code is protected by copyright law and international treaties. This source code is made available to You subject to the terms and conditions of the Software License for the webMUSHRA.js Software. Said terms and conditions have been made available to You prior to Your download of this source code. By downloading this source code You agree to be bound by the above mentionend terms and conditions, which can also be found here: https://www.audiolabs-erlangen.de/resources/webMUSHRA. Any unauthorised use of this source code may result in severe civil and criminal penalties, and will be prosecuted to the maximum extent possible under law. 

**************************************************************************/

function LikertScale(_responseConfig, _prefix, _initDisabled, _callback) {
  this.responseConfig = _responseConfig;
  this.prefix = _prefix;
  this.initDisabled = _initDisabled;
  this.callback = _callback;  
  this.group = null;
  this.elements = null;
} 

LikertScale.prototype.enable = function () {
  $('input[name='+this.prefix+'_response]').checkboxradio('enable');
}

LikertScale.prototype.render = function (_parent) {
  this.group = $("<fieldset data-role='controlgroup' data-type='horizontal'></fieldset>");
  _parent.append(this.group);
  this.elements = [];
  var i;
  if(this.responseConfig != null){
    for (i = 0; i < this.responseConfig.length; ++i) {
      var responseValueConfig = this.responseConfig[i];
      var img = "";
      if (responseValueConfig.title){
        var header = $("<p class='session-header'>" + responseValueConfig.title+ "</div>");
        this.group.append(header)
        if (responseValueConfig.description){
          var description = $("<p class='session-description'>" + responseValueConfig.description+"</p></br>")
          this.group.append(description);
        }
      }
      if (responseValueConfig.type=="rate"){
        var element = $("<p>" + responseValueConfig.label + "</p>" );
        this.group.append(element);
        if (responseValueConfig.rate){
          var rates = [];
          for (var i = 0; i < responseValueConfig.rate; ++i) {
              rates.push(i);
          }
          var labels = [];
          for (var i = 0; i < rates.length; ++i) {
            labels.push(rates[i].toString());
          }
        
          this.group.append(element);

          // 创建一个容器来包裹所有按钮和标签
          var container = $("<div class='radio-container'></div>");
          var labelLeft = $("<label class='label-left'>"+responseValueConfig.min_value+"</label>");
          var labelRight = $("<label class='label-right'>"+responseValueConfig.max_value+"</label>");
          container.append(labelLeft);

          for (var i = 0; i < rates.length; ++i) {
            img = "<img id='"+this.prefix+"_response_img_"+i+"' width='50%' src='"+responseValueConfig.img+"'/>";
            var radio = $("<div class = 'ui-radio-1'><input type='radio' class='custom1' data-mini='false' value='"+rates[i]+"' name='"+this.prefix+"_response' id='"+this.prefix+"_response_"+i+"'/>");
            var label = $("<label for='"+this.prefix+"_response_"+i+"' class='custom-label'><center>"+labels[i]+"<br/>"+img+"</center></label></div>");
            container.append(radio, label);
            this.elements.push(radio, label);
          }

          container.append(labelRight);
          this.group.append(container);
        }
        this.group.append($("</br>"));
      }else if(responseValueConfig.type == "rate2"){
        var element = $("<p>" + responseValueConfig.label + "</p>" );
        this.group.append(element);
        if (responseValueConfig.rate){
          var rates = [];
          for (var i = 0; i < responseValueConfig.rate; ++i) {
              rates.push(i + 1); // count from 1
          }
          var labels = [];
          if (responseValueConfig.rate_labels) {
            labels = responseValueConfig.rate_labels;
          } else {
            for (var i = 0; i < rates.length; ++i) {
              labels.push("选项 " + rates[i]);
            }
          }
        
          
          var container = $("<div class='rate2-container'></div>");

          for (var i = 0; i < rates.length; ++i) {
            var optionContainer = $("<div class='rate2-option'></div>");
            

            // <input type="radio" name="gender" value="male" checked></input>
            var radio = $(
              "<input type='radio' class='rate2-radio' value='"+(i+1)+"' " +
                    "name='"+this.prefix+"_response' " +
                    "id='"+this.prefix+"_response_"+i+"'>&nbsp;" +
              "<span class='radio-text'>&nbsp&nbsp"+(i+1)+" = "+labels[i]+"</span>"
            );
            
            optionContainer.append(radio);
            container.append(optionContainer);
            this.elements.push(radio);
          }

          this.group.append(container);
        }
        // this.group.append($("</br>"));
      }else if(responseValueConfig.type == "select"){
        if(responseValueConfig.label){
          var element = $("<p>" + responseValueConfig.label + "</p>" );
          this.group.append(element);
        }
        
        if (responseValueConfig.select_value){
          var select_value = responseValueConfig.select_value.split('/')
        }
        var img = "";
        var container = $("<div class='radio-container'></div>");
        
        for (var i = 0; i < select_value.length; ++i) {
          
          if (responseValueConfig.img) {
            img = "<br/><img id='"+this.prefix+"_response_img_"+i+"' width='20%' src='"+responseValueConfig.img+"'/>";
          }
          var radio = $("<input type='radio' class='custom2' data-mini='false' value='"+i+"' name='"+this.prefix+"_response' id='"+this.prefix+"_response_"+i+"'/>");
          var label = $("<label for='"+this.prefix+"_response_"+i+"' class='custom-label2'><center>"+select_value[i]+img+"</center></label></div>");
          container.append(radio, label);
          this.elements.push(radio, label);
        }
        this.group.append(container);
        this.group.append($('</br>'));

      }else if(responseValueConfig.type == "option"){
        var element = $("<p>" + responseValueConfig.label + "</p>" );
        this.group.append(element);
        var element = $("<input id='"+this.prefix+"_comment_option'></input>");
        this.group.append(element);
        this.callback(this.prefix);
      }else if(responseValueConfig.type == "text"){
        var element = $("<p>" + responseValueConfig.label + "</p>" );
        this.group.append(element);
        var textInput = $("<input type='text' id='"+this.prefix+"_text_input' required></input>");
        this.group.append(textInput);
      }else{ 
        for (i = 0; i < this.responseConfig.length; ++i) {
          var responseValueConfig = this.responseConfig[i];
          var img = "";
          if (responseValueConfig.img) {
            img = "<img id='"+this.prefix+"_response_img_"+i+"' src='"+responseValueConfig.img+"'/><br/>";
          }
          var element = $("<input type='radio' data-mini='false' value='"+responseValueConfig.value+"' name='"+this.prefix+"_response' id='"+this.prefix+"_response_"+i+"'/> \
            <label for='"+this.prefix+"_response_"+i+"'><center>"+img+""+responseValueConfig.label+"</center></label> \
          ");
          this.elements[this.elements.length] = element;
          this.group.append(element);
        }
      }
      
      
    }

    this.group.change((function() {
      var set = false;
      for (i = 0; i < this.elements.length; ++i) {
        if (set === true) {
          $("#"+this.prefix+"_response_img_"+i).attr("src", this.responseConfig[0].img);
        } else {
          if ($("#"+this.prefix+"_response_"+i+":checked").val()) {
            set = true;
            $("#"+this.prefix+"_response_img_"+i).attr("src", this.responseConfig[0].imgSelected);
          } else {
            if (this.responseConfig[0].type != "select"){
              $("#"+this.prefix+"_response_img_"+i).attr("src", this.responseConfig[0].imgHigherResponseSelected);
            }else{
              $("#"+this.prefix+"_response_img_"+i).attr("src", this.responseConfig[0].img);
            }
            
            
          }
        }

      }

      if (this.callback) {
        
        this.callback(this.prefix);
      }
    }).bind(this));
  }
  if (this.initDisabled) {
    this.group.children().attr('disabled', 'disabled');    
  }
  
};

