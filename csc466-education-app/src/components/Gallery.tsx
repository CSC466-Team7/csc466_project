import React from 'react';
import { createStyles, makeStyles, Theme } from '@material-ui/core/styles';

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    gallery: {
      display: 'flex',
      justifyContent: 'space-around',
    },
  }),
);

interface GalleryProps {
  children: React.ReactNode,
}

export default function Gallery(props: GalleryProps) {
  const classes = useStyles();

  return (
    <div className={classes.gallery}>
      {props.children}
    </div>
  );
}
