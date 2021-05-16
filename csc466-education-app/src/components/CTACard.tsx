import React from "react";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    card: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      margin: "60px auto",
      width: "950px",
      padding: "8px 25px 0",
      backgroundColor: "#eeeef0",
      borderRadius: "4px",
    },
  }),
);

interface CTAProps {
  children: React.ReactNode,
}

export default function CTACard(props: CTAProps) {
  const classes = useStyles();

  return (
    <div className={classes.card}>
      {props.children}
    </div>
  );
}
