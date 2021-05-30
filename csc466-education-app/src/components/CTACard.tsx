import { Button, Paper } from "@material-ui/core";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import { KeyboardArrowRightOutlined } from "@material-ui/icons";
import React, { useState } from "react";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    card: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      flexWrap: "wrap",
      margin: "25px auto",
      width: "80%",
      padding: "12px 20px",
      backgroundColor: theme.palette.background.paper
    }
  })
);

interface CTAProps {
  title: string,
  description: string,
  buttonText: string,
  linkTo: string,
  secondary? : boolean
}

export default function CTACard(props: CTAProps) {
  const [hovering, setHovering] = useState(false);
  const classes = useStyles();
  
  return (
    <Paper
      className={classes.card}
      elevation={hovering ? 24 : 6}
      variant={props.secondary ? "outlined" : "elevation"}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}>
      <div>
        <h2>{props.title}</h2>
        <p>{props.description}</p>
      </div>
      <Button
        component="button"
        color="primary"
        variant="contained"
        href={props.linkTo}
      >
        {props.buttonText}
        {(hovering || props.secondary) && <KeyboardArrowRightOutlined/>}
      </Button>
    </Paper>
  );
}
