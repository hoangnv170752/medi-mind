const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const communicationSchema = new Schema({
    email:{
        type: String ,
    },
    message:{
        type: String ,
        required: true
    },
    from: {
        type: String,
        required: true
    },
    to: {
        type: String,
    }
}, {
    timestamps: true,
});

const Communication = mongoose.model("Communication", communicationSchema, "communications");
module.exports = Communication;
