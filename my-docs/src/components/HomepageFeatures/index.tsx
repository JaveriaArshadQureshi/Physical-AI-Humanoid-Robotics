import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  image: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Future-Ready Learning',
    image: '/img/future.png',
    description: (
      <>
        Our curriculum is designed for the future of robotics, incorporating cutting-edge research and emerging technologies in Physical AI.
      </>
    ),
  },
  {
    title: 'Rocket-Powered Progress',
    image: '/img/rocket.png',
    description: (
      <>
        Accelerate your understanding with our step-by-step approach that propels you from beginner to expert in humanoid robotics.
      </>
    ),
  },
  {
    title: 'Advanced Robotics',
    image: '/img/future.png',
    description: (
      <>
        Master the technologies that will shape the future of human-robot interaction and autonomous systems.
      </>
    ),
  },
];

function Feature({title, image, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img src={image} className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
