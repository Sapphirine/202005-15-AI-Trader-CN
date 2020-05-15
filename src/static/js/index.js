/**
 * Created by KiraMelody on 2017/12/26.
 */

$(function () {

   $("#prediction").on('click',function () {
       func = "prediction"
       $("#choose_function").html("<span></span>输入预测<span class='caret'></span>" +
           "<input type='text' name='function' hidden='true' id='refunction' value='prediction'>");
   });
   $("#grammer").on('click',function () {
       func = "grammer"
       $("#choose_function").html("<span></span>句法比较<span class='caret'></span>" +
           "<input type='text' name='function' hidden='true' id='refunction' value='grammer'>");
   });
   $("#pinyin").on('click',function () {
       func = "pinyin"
       $("#choose_function").html("<span></span>拼音汉字<span class='caret'></span>" +
           "<input type='text' name='function' hidden='true' id='refunction' value='pinyin'>");
   });
   $("#chinese").on('click',function () {
       func = "chinese"
       $("#choose_function").html("<span></span>中文输入<span class='caret'></span>" +
           "<input type='text' name='function' hidden='true' id='refunction' value='chinese'>");
   });
   $("#select-item1").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item1").html());
       $("#func-search").click();
   });
   $("#select-item2").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item2").html());
       $("#func-search").click();
   });
   $("#select-item3").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item3").html());
       $("#func-search").click();
   });
   $("#select-item4").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item4").html());
       $("#func-search").click();
   });
   $("#select-item5").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item5").html());
       $("#func-search").click();
   });
   $("#select-item6").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item6").html());
       $("#func-search").click();
   });
   $("#select-item7").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item7").html());
       $("#func-search").click();
   });
   $("#select-item8").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item8").html());
       $("#func-search").click();
   });
   $("#select-item9").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item9").html());
       $("#func-search").click();
   });
   $("#select-item10").on('click',function () {
       str = $("#search-query").val();
       $("#search-query").val(str + $("#select-item10").html());
       $("#func-search").click();
   });
});

$(document).ready(function () {
    str = $("#choose_function").html();
    str = str.replace(/\s+/g,"")[13];
    if (str == "句") {
        func = "grammer";
    } else if (str == "拼") {
        func = "pinyin";
    } else if (str == "中") {
        func = "chinese";
    }
    $("#refunction").val(func);
});

var func = "prediction";