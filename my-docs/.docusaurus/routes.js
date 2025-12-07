import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/markdown-page',
    component: ComponentCreator('/markdown-page', '3d7'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'b40'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '77a'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'fef'),
            routes: [
              {
                path: '/docs/autonomous-navigation-humanoids',
                component: ComponentCreator('/docs/autonomous-navigation-humanoids', '8d2'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/building-humanoid-actuators',
                component: ComponentCreator('/docs/building-humanoid-actuators', 'f0c'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/capstone-project-guide',
                component: ComponentCreator('/docs/capstone-project-guide', '3d8'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/control-systems',
                component: ComponentCreator('/docs/control-systems', '559'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/gazebo-simulation',
                component: ComponentCreator('/docs/gazebo-simulation', '061'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/imitation-learning-teleoperation',
                component: ComponentCreator('/docs/imitation-learning-teleoperation', '442'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', 'cda'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/introduction-to-physical-ai',
                component: ComponentCreator('/docs/introduction-to-physical-ai', 'f50'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/kinematics-dynamics',
                component: ComponentCreator('/docs/kinematics-dynamics', '44f'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/linux-ros2-foundations',
                component: ComponentCreator('/docs/linux-ros2-foundations', '811'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/nvidia-isaac-sim',
                component: ComponentCreator('/docs/nvidia-isaac-sim', 'ef7'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/real-robot-control-architecture',
                component: ComponentCreator('/docs/real-robot-control-architecture', 'd67'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/reinforcement-learning-robotics',
                component: ComponentCreator('/docs/reinforcement-learning-robotics', '8d2'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/robot-perception',
                component: ComponentCreator('/docs/robot-perception', '7ee'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/safety-edge-computing',
                component: ComponentCreator('/docs/safety-edge-computing', 'd02'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/sensor-fusion-localization',
                component: ComponentCreator('/docs/sensor-fusion-localization', 'e06'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/docs/vision-language-action-models',
                component: ComponentCreator('/docs/vision-language-action-models', '8a9'),
                exact: true,
                sidebar: "bookSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', 'e5f'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
