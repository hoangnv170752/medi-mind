const express = require("express");
const router = express.Router();
const User = require("../models/user");
const ContactUs = require("../models/contactUs");
const Communication = require("../models/communication");
router.post("/add-contact-us", async (req, res) => {
  const { name, phone, email, message } = req.body;

  try {
    const newContactUs = new ContactUs({
      name,
      phone,
      email,
      message,
    });

    const savedContactUs = await newContactUs.save();

    res.status(200).json(savedContactUs);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get("/get-users" , async (req ,res) =>{

  try {
    const findUser =  await User.find();
    if(!findUser) res.json("No user found");
    res
    .status(200)
    .json(findUser);
  } catch (error) {
    
  }

} );

router.put("/profile-update", async (req, res) => {
  const { userId, updatedProfile } = req.body;
  try {
    const updatedUser = await User.findByIdAndUpdate(
      userId,
      { $set: updatedProfile },
      { new: true, runValidators: true }
    );

    res.status(200).json({ status: "Success", user: updatedUser });
  } catch (error) {
    console.error("Error updating profile:", error.message);
  }
});

router.get("/get-medications/:userEmail", async (req, res) => {
  const userEmail = req.params.userEmail;
  try {
    const user = await User.findOne({ email: userEmail });
    if (user) {
      if (user.medicalHistory && user.medicalHistory.length > 0) {
        res.status(200).json(user.medicalHistory);
      } else {
        res.status(200).json([]);
      }
    } else {
      res.status(404).json({ error: "User not found" });
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Server Error" });
  }
});

router.post('/add-medications/:userEmail', async (req, res) => {
  try {
    const { userEmail } = req.params;
    const { name, dosage, frequency } = req.body;


    const user = await User.findOne({ email: userEmail });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    user.medicalHistory.push({
      medications: [{ name, dosage, frequency }],
    });

    await user.save();

    res.status(201).json(user);
  } catch (err) {
    console.error(err);
    res.status(500).send('Server Error');
  }
});

router.post("/add-message-chatbot" , async (req , res) => {

  const{email, message ,from, to} = req.body ;

  const newEntry = await new Communication({email, message ,from, to: "chatbot@gmail.com"});

  try {
    await newEntry.save();
    res.status(200).json({message: "Successfully sent"});
  } catch (error) {
    res.json({message: "couldnt sent the message"});
  }
});

router.get("/get-message/:email" , async (req , res) => {

  const email = req.params.email;
  const emailSent = req.query.receiver;
  console.log(email);

  try {
    const message = await Communication.find({email: email});
    console.log(message.length);

    if (emailSent) {
      message.filter((res) => res.from !== emailSent || res.to === emailSent);
      console.log(message.length);
    }
    res.json(message);
  } catch (error) {
    res.status(500).json({ error: "Could not get the message" }); 
  }
});


module.exports = router;
