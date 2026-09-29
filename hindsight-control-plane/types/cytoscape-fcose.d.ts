declare module 'cytoscape-fcose' {
  import { Ext } from 'cytoscape';

  interface FcoseLayoutOptions {
    name: 'fcose';
    quality?: 'default' | 'draft' | 'proof';
    randomize?: boolean;
    animate?: boolean;
    animationDuration?: number;
    animationEasing?: string;
    fit?: boolean;
    padding?: number;
    nodeDimensionsIncludeLabels?: boolean;
    uniformNodeDimensions?: boolean;
    packComponents?: boolean;
    step?: 'transformed' | 'untransformed' | 'all';
    samplingType?: boolean;
    sampleSize?: number;
    nodeSeparation?: number;
    piTol?: number;
    nodeRepulsion?: number;
    idealEdgeLength?: number;
    edgeElasticity?: number;
    nestingFactor?: number;
    gravity?: number;
    numIter?: number;
    initialTemp?: number;
    coolingFactor?: number;
    minTemp?: number;
    fixedNodeConstraint?: any[];
    alignmentConstraint?: any[];
    relativePlacementConstraint?: any[];
  }

  const fcose: Ext;
  export = fcose;
}

// Extend cytoscape module declarations to include missing CSS properties and methods
declare module 'cytoscape' {
  // Add the main cytoscape function
  interface CytoscapeOptions {
    container?: HTMLElement;
    elements?: any[];
    style?: any[];
    layout?: any;
    selectionType?: string;
    userZoomingEnabled?: boolean;
    userPanningEnabled?: boolean;
    boxSelectionEnabled?: boolean;
    [key: string]: any;
  }

  // Add Core interface
  interface Core {
    add(elements: any): void;
    layout(options: any): any;
    on(event: string, selector: string, handler: Function): void;
    on(event: string, handler: Function): void;
    off(event: string, handler?: Function): void;
    removeListener(event: string, handler?: Function): void;
    destroy(): void;
    nodes(): any;
    edges(): any;
    elements(): any;
    getElementById(id: string): any;
    fit(): void;
    zoom(): number;
    zoom(level: number): void;
    pan(): { x: number; y: number };
    pan(position: { x: number; y: number }): void;
    resize(): void;
    animate(options: any, timing?: any): any;
  }

  // Define cytoscape as both callable function and object with properties
  interface CytoscapeStatic {
    (options: CytoscapeOptions): Core;
    use(extension: any): void;
  }

  // Make cytoscape the default export
  const cytoscape: CytoscapeStatic;
  export = cytoscape;

  namespace Css {
    interface Node {
      'target-arrow-color'?: string;
      'target-arrow-shape'?: string;
      'target-arrow-size'?: number;
      'curve-style'?: string;
      'text-valign'?: string;
      'text-halign'?: string;
      'font-size'?: string;
      'font-weight'?: string | number;
      'text-margin-y'?: number;
      'text-wrap'?: string;
      'text-max-width'?: string;
      'text-background-color'?: string;
      'text-background-opacity'?: number;
      'text-background-padding'?: string;
      'text-background-shape'?: string;
      'border-width'?: number;
      'border-color'?: string;
      'border-opacity'?: number;
      'background-color'?: string;
      'line-color'?: string;
      'overlay-opacity'?: number;
      'overlay-color'?: string;
      'overlay-padding'?: number;
      'z-index'?: number;
    }

    interface Edge {
      'target-arrow-color'?: string;
      'target-arrow-shape'?: string;
      'target-arrow-size'?: number;
      'curve-style'?: string;
      'line-color'?: string;
      'overlay-opacity'?: number;
      'overlay-color'?: string;
      'overlay-padding'?: number;
      'z-index'?: number;
    }
  }
}