import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Book sidebar for Physical AI & Humanoid Robotics
  bookSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Foundational Chapters',
      items: [
        'introduction-to-physical-ai',
        'linux-ros2-foundations',
        'gazebo-simulation',
        'nvidia-isaac-sim',
        'real-robot-control-architecture',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Technical Chapters',
      items: [
        'sensor-fusion-localization',
        'kinematics-dynamics',
        'control-systems',
        'robot-perception',
        'vision-language-action-models',
        'reinforcement-learning-robotics',
        'imitation-learning-teleoperation',
      ],
    },
    {
      type: 'category',
      label: 'Implementation and Specialized Topics',
      items: [
        'building-humanoid-actuators',
        'autonomous-navigation-humanoids',
        'safety-edge-computing',
        'capstone-project-guide',
      ],
    },
  ],
};

export default sidebars;
