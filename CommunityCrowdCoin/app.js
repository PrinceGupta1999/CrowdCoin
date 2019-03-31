var express       = require("express"),
    app           = express(),
    bodyParser    = require("body-parser"),
    mongoose      = require("mongoose");
mongoose.connect("mongodb://localhost/crowdcoin",{ useNewUrlParser: true });
const commentSchema = new mongoose.Schema({
    text: String,
    upvotes: Number,
    downvotes: Number,
    author: String,
    campaign: String
});
const model = mongoose.model("Comment", commentSchema);
app.use(bodyParser.urlencoded({extended: true}));
app.set("view engine", "ejs");
app.use(express.static(__dirname + "/public"));
app.get("/", function(req, res) {

    model.find({}, (err, data) => {
      if(!err)
      {
        res.render("index.ejs", {comments : data, campaign: req.query});
      }
      else {
           throw err("Error while fetching");
         }
    });
});
app.post('/', (req, res) => {
    console.log(req.body);
    model.create({text: req.body.text, author: req.body.author, campaign: req.body.name, upvotes: 0, downvotes: 0}, (err, response) => {
        if(!err)
        {
            serialize = function(obj) {
              var str = [];
              for (var p in obj)
                if (obj.hasOwnProperty(p)) {
                  str.push(encodeURIComponent(p) + "=" + encodeURIComponent(obj[p]));
                }
              return str.join("&");
            }
            console.log(serialize({name: req.body.name, description: req.body.description}));
          res.redirect("/?" + serialize({name: req.body.name, description: req.body.description}));
        }
        else {
          throw err("Error while inserting data");
        }
    });

});
app.listen(8000, process.env.IP, function(){
   console.log("community.crowdcoin.com server has started");
});
