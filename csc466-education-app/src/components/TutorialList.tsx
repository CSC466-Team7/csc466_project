import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import React from "react";
import CustomCard from "./CustomCard";
import Gallery from "./Gallery";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    gallery: {
      display: "flex",
      justifyContent: "space-around",
      flexWrap: "wrap"
    }
  })
);

interface TutorialProps {
}

export default function TutorialList(props: TutorialProps) {
  const classes = useStyles();
  
  const cardDetails = {
    img: {
      url: "https://miro.medium.com/max/781/1*fGX0_gacojVa6-njlCrWZw.png",
      title: "Decision Tree Visual"
    },
    content: {
      title: "Decision Tree Algorithm",
      description:
        "Learn how to implement a decision tree from scratch with Python"
    },
    linkTo: "/#/example/1"
  };
  return (
    <Gallery>
      <CustomCard
        {...cardDetails}/>
      <CustomCard
        {...cardDetails}/>
      <CustomCard
        {...cardDetails}/>
    </Gallery>
  );
}
