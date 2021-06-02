import React from "react";
import { Link } from "react-router-dom";
import CTACard from "../components/CTACard";

import Gallery from "../components/Gallery";
import Header from "../components/Header";
import { tutorials } from "../data";

export default function Tutorials() {
  const ex1 = {
    url: "https://cdn.xxl.thumbs.canstockphoto.com/number-1-sign-design-template-element-black-icon-on-transparent-background-vector-clip-art_csp44589215.jpg",
    title: "the number 1"
  };
  
  const ex2 = {
    url: "https://snapchatemojis.com/wp-content/uploads/2015/09/2-fingers.png",
    title: "the number 2"
  };
  
  return (
    <>
      <Header
        title={"Tutorials"}
        description={"Get into the learn by doing spirit with these walkthrough tutorials."}
      />
      
      <Gallery cards={tutorials}/>
      
      <Header
        description={"Check the following out if you're having trouble working through the\n" +
        "        tutorials"}
        title={"Resources"}/>
      <CTACard
        title={"Unfamiliar with decision trees?"}
        description={"Learn more about them by reading this article"}
        buttonText={"Introduction to decision trees"}
        linkTo={"/introduction"}
        secondary={true}/>
      <CTACard
        title="Not comfortable with Pandas, Numpy, or Sklearn?"
        description="Refresh your preliminary skills with easy to follow guides"
        buttonText="Preliminary skills"
        linkTo="/preliminary-skills"
        secondary={true}
      />
    </>
  );
}
