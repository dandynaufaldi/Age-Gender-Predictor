var lines = [];
var result = [];
var headers = undefined;
var lastIdx = 0;
// var nextbut = document.getElementById('next-but');
//      nextbut.addEventListener("click", saveCurrent(showNext, lastIdx));
$(document).ready(function() {
     $.ajax({
        type: "GET",
        url: "http://127.0.0.1:8887/Age-Gender-Predictor/crawl/indon.csv",
        dataType: "text",
        success: function(data) {
                    processData(data, lines);
                    $.ajax({
                        type: "GET",
                        url: "http://127.0.0.1:8887/Age-Gender-Predictor/crawl/hasil.csv",
                        dataType: "text",
                        success: function(data) {
                                    processData(data, result);
                                    lastIdx = result.length;
                                    // showNext(lastIdx);
                                    saveCurrent(showNext, lastIdx);
                                },
                        error: function(err){
                                // showNext(lastIdx);
                                saveCurrent(showNext, lastIdx);
                        }
                     });
                }
     });
    //  var nextbut = document.getElementById('next-but');
    //  nextbut.addEventListener("click", saveCurrent(showNext, lastIdx));
});

function exportToCsv(filename, rows) {
    var processRow = function (row) {
        var finalVal = '';
        for (var j = 0; j < row.length; j++) {
            var innerValue = row[j] === null ? '' : row[j].toString();
            if (row[j] instanceof Date) {
                innerValue = row[j].toLocaleString();
            };
            var result = innerValue.replace(/"/g, '""');
            if (result.search(/("|,|\n)/g) >= 0)
                result = '"' + result + '"';
            if (j > 0)
                finalVal += ',';
            finalVal += result;
        }
        return finalVal + '\n';
    };

    var csvFile = '';
    csvFile += processRow(headers);
    for (var i = 0; i < rows.length; i++) {
        csvFile += processRow(rows[i]);
    }

    var blob = new Blob([csvFile], { type: 'text/csv;charset=utf-8;' });
    if (navigator.msSaveBlob) { // IE 10+
        navigator.msSaveBlob(blob, filename);
    } else {
        var link = document.createElement("a");
        if (link.download !== undefined) { // feature detection
            // Browsers that support HTML5 download attribute
            var url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
}

function processData(allText, array) {
    console.log('read data')
    var allTextLines = allText.split(/\r\n|\n/);
    headers = allTextLines[0].split(',');

    for (let i=1; i<allTextLines.length; i++) {
        var data = allTextLines[i].split(',');
        if (data.length == headers.length) {
            var tarr = [];
            for (let j=0; j<headers.length; j++) {
                tarr.push(data[j]);
            }
            array.push(tarr);
        }
    }
}

function saveCurrent(showNext, last_idx){
    idxlist = document.getElementsByClassName('idx');
    if (idxlist.length){
        for (let i = 0; i < idxlist.length; i++){
            let idx = parseInt(idxlist[i].innerHTML);
            lastIdx = idx + 1;
            while (idx > result.length){
                result.push(lines[result.length]);
            }
            if (idx == result.length){
                let temp = []
                temp.push(lines[idx][0]);
                temp.push(lines[idx][1]);
                let age = idxlist[i].nextSibling.nextSibling.firstElementChild.value;
                let gender = idxlist[i].nextSibling.nextSibling.nextSibling.firstElementChild.value;
                temp.push(age);
                temp.push(gender);
                result.push(temp);
            }
        }
    }
    showNext(lastIdx);
}

function downloadCur(){
    idxlist = document.getElementsByClassName('idx');
    if (idxlist.length){
        for (let i = 0; i < idxlist.length; i++){
            let idx = parseInt(idxlist[i].innerHTML);
            lastIdx = idx + 1;
            while (idx > result.length){
                result.push(lines[result.length]);
            }
            if (idx == result.length){
                let temp = []
                temp.push(lines[idx][0]);
                temp.push(lines[idx][1]);
                let age = idxlist[i].nextSibling.nextSibling.firstElementChild.value;
                let gender = idxlist[i].nextSibling.nextSibling.nextSibling.firstElementChild.value;
                temp.push(age);
                temp.push(gender);
                result.push(temp);
            }
        }
    }
    exportToCsv('hasil.csv', result);
    // const rows = [["name1", "city1", "some other info"], ["name2", "city2", "more info"]];
    // let csvContent = "data:text/csv;charset=utf-8,";
    // csvContent += headers.join(",") + "\r\n";
    // result.forEach(function(rowArray){
    //     let row = rowArray.join(",");
    //     csvContent += row + "\r\n";
    //     }); 
    // var encodedUri = encodeURI(csvContent);
    // var link = document.createElement("a");
    // link.setAttribute("href", encodedUri);
    // link.setAttribute("download", "wiki_result.csv");
    // link.innerHTML= "Click Here to download";
    // document.body.appendChild(link); // Required for FF
    // link.click();
    // document.body.removeChild(link);
}

function showNext(last_idx){
    var isi = "";
    var counter = 0;
    for (let i = last_idx; i < lines.length && counter < 20; i++, counter++) {
        let current = lines[i];
        let age = parseInt(current[1])
        let gender = parseFloat(current[2]);
        let imgname =current[0];
        var row = "<tr>";
        
        row += '<td class="idx">' + i.toString() + '</td>';
        row += '<td><img src="' + imgname + '"></td>';
        row += '<td><input type="text" value="' + age.toString() + '"></td>';
        row += '<td><input type="text" value="' + gender.toString() + '"></td>';
        row += '</tr>'

        isi += row
    }
    tabel = document.getElementById('tabel-body');
    tabel.innerHTML = isi;
    window.scrollTo(0,0);
}