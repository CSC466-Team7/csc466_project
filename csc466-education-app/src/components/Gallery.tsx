import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import React from "react";
import CustomCard, { CardProps } from "./CustomCard";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    gallery: {
      display: "flex",
      justifyContent: "space-around",
      flexWrap: "wrap"
    }
  })
);

interface GalleryProps {
  cards: CardProps[]
}

export default function Gallery(props: GalleryProps) {
  const classes = useStyles();
  
  return (
    <div className={classes.gallery}>
      {props.cards.map((skill =>
        <CustomCard
          {...skill}
          key={skill.content.title}/>
      ))}
    </div>
  );
}
