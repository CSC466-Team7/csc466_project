import { IconButton, Paper } from "@material-ui/core";
import { FileCopy } from "@material-ui/icons";
import AddIcon from "@material-ui/icons/Add";
import RemoveIcon from "@material-ui/icons/Remove";
import React, { useEffect, useState } from "react";
import { Collapse } from "react-bootstrap";
import { CopyToClipboard } from "react-copy-to-clipboard";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import dark from "react-syntax-highlighter/dist/esm/styles/prism/material-dark";
// @ts-ignore
import rehypeRaw from "rehype-raw";
import { ASSETS_URL } from "../constants";

function OutputComponent(props: any) {
  const [collapsed, setCollapsed] = useState(true);
  
  const changeCollapsed = () => {
    setCollapsed(!collapsed);
  };
  
  return (
    <>
      <IconButton style={{float: "right", color: "#37474f"}}
        onClick={changeCollapsed}>
        {!collapsed ?
          <AddIcon/> :
          <RemoveIcon/>
        }
      </IconButton>
      
      <Collapse in={collapsed}>
        <div>
          <code className={props.className}
            children={props.children} {...props} />
        </div>
      </Collapse>
    </>
  );
}

export default function Markdown(props: { fileName: string; }) {
  const [markdownAsString, setMarkdownAsString] = useState("");
  const fetchMarkdown = async () => {
    const url = `${ASSETS_URL}/${props.fileName}`;
    // Switch when testing locally
    // const file = await
    // import("../markdown/heart_decision_tree_classifier.md"); const toFetch =
    // file.default;
    const toFetch = url;
    const res = await fetch(toFetch);
    const resText = await res.text();
    setMarkdownAsString(resText);
  };
  useEffect(() => {
    fetchMarkdown();
  }, []);
  const components: any = {
    // @ts-ignore
    code({node, inline, className, children, ...props}) {
      const match = /language-(\w+)/.exec(className || "");
      return !inline && match ? (
        <span>
          <CopyToClipboard text={String(children)}>
            <IconButton color="primary"
              component="span"
              style={{float: "right", marginLeft: 10}}>
              <FileCopy/>
            </IconButton>
          </CopyToClipboard>
          <SyntaxHighlighter style={dark} language={match[1]} PreTag="div"
            children={
              String(children)
                .replace(/\n$/, "")
            }
            {...props} />
        </span>
      
      ) : (
        <OutputComponent className={className} children={children} {...props} />
      );
    }
  };
  return (
    <Paper elevation={3} style={{padding: 20, margin: 20}}>
      <article className="markdown-body">
        <ReactMarkdown
          components={components}
          skipHtml={false}
          rehypePlugins={[rehypeRaw]}>{markdownAsString}</ReactMarkdown>
      </article>
    </Paper>
  );
}