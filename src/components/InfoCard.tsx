import { Component, JSX } from "solid-js";
import { colors } from "../styles/colors";
import { css } from "@emotion/css";

const InfoCard: Component<{ icon: JSX.Element, title: string, description: string, value: string }> = ({ icon, title, description, value }) => (
    <div class={css`text-align: center;`}>
      {icon}
      <h3 class={css`color: ${colors.primary}; margin: 12px 0; font-size: 24px;`}>{title}</h3>
      <p class={css`font-size: 18px; margin-bottom: 8px;`}>{description}</p>
      <span class={css`font-size: 36px; font-weight: 700; color: ${colors.secondary};`}>{value}</span>
    </div>
  );

  export default InfoCard;